#include "tileoperator.hpp"
#include "img_utils.hpp"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

struct SpatialMetricsAccum {
    int32_t K = 0;
    std::vector<uint64_t> area;
    std::vector<uint64_t> perim;
    std::vector<uint64_t> shared_ij;

    explicit SpatialMetricsAccum(int32_t K_)
        : K(K_),
          area(static_cast<size_t>(K_) + 1, 0),
          perim(static_cast<size_t>(K_) + 1, 0),
          shared_ij(static_cast<size_t>(K_ + 1) * static_cast<size_t>(K_) / 2, 0) {}

    size_t triIndex(int32_t i, int32_t j) const {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(2 * K - i + 1) / 2;
        return base + static_cast<size_t>(j - i - 1);
    }

    void add(const SpatialMetricsAccum& other) {
        for (size_t i = 0; i < area.size(); ++i) area[i] += other.area[i];
        for (size_t i = 0; i < perim.size(); ++i) perim[i] += other.perim[i];
        for (size_t i = 0; i < shared_ij.size(); ++i) shared_ij[i] += other.shared_ij[i];
    }
};

inline void spatialAccumulateEdge(SpatialMetricsAccum& m, uint8_t a, uint8_t b) {
    if (a == b) return;
    int32_t i = static_cast<int32_t>(a);
    int32_t j = static_cast<int32_t>(b);
    if (i > j) std::swap(i, j);
    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(2 * m.K - i + 1) / 2;
    m.shared_ij[base + static_cast<size_t>(j - i - 1)]++;
    m.perim[static_cast<size_t>(a)]++;
    m.perim[static_cast<size_t>(b)]++;
}

void rescaleGeometryCoordinates(nlohmann::json& node, double scaleInv) {
    if (!node.is_array()) {
        return;
    }
    if (node.size() == 2 && node[0].is_number() && node[1].is_number()) {
        node[0] = node[0].get<double>() * scaleInv;
        node[1] = node[1].get<double>() * scaleInv;
        return;
    }
    for (auto& child : node) {
        rescaleGeometryCoordinates(child, scaleInv);
    }
}

} // namespace

void TileOperator::smoothTopLabels2D(const std::string& outPrefix, int32_t islandSmoothRounds, bool fillEmptyIslands) {
    requireNoFeatureIndex(__func__);
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (islandSmoothRounds <= 0) {
        error("%s: islandSmoothRounds must be > 0", __func__);
    }
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    // readNextRecord2DAsPixel() yields pixel-space coordinates for:
    // 1) float-coordinate inputs, and
    // 2) int-coordinate inputs marked as scaled (mode_ & 0x2).
    const bool recCoordsInPixel = ((mode_ & 0x4) == 0) || ((mode_ & 0x2) != 0);
    std::string outFile = outPrefix + ".bin";
    std::string outIndex = outPrefix + ".index";
    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) {
        error("%s: Cannot open output file %s", __func__, outFile.c_str());
    }
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        close(fdMain);
        error("%s: Cannot open output index %s", __func__, outIndex.c_str());
    }

    IndexHeader idxHeader = formatInfo_;
    std::vector<uint32_t> outKvec{1};
    idxHeader.packKvec(outKvec);
    idxHeader.mode = (K_ << 16) | (recCoordsInPixel ? 0x2u : 0x0u) | 0x5u;
    idxHeader.pixelResolution = getRasterPixelResolution();
    idxHeader.recordSize = sizeof(int32_t) * 2 + sizeof(int32_t) + sizeof(float);
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        close(fdMain);
        close(fdIndex);
        error("%s: Failed writing index header", __func__);
    }

    struct SmoothTileResult {
        std::vector<char> data;
        uint32_t nOut = 0;
        size_t nCollisionIgnored = 0;
        size_t nOutOfRangeIgnored = 0;
        int32_t row = 0;
        int32_t col = 0;
    };

    const bool useParallel = (threads_ > 1 && blocks_.size() > 1);
    const size_t recSize = static_cast<size_t>(idxHeader.recordSize);
    const size_t chunkTileCount = useParallel
        ? std::max<size_t>(static_cast<size_t>(threads_) * 4, 1)
        : static_cast<size_t>(1);

    auto processTile = [&](size_t bi, std::ifstream& in, SmoothTileResult& out) {
        const TileInfo& blk = blocks_[bi];
        TileGeom geom;
        initTileGeom(blk, geom);
        out.row = geom.key.row;
        out.col = geom.key.col;
        const size_t width = geom.W;
        const size_t height = geom.H;
        if (height > 0 && width > std::numeric_limits<size_t>::max() / height) {
            error("%s: Raster size overflow in tile (%d, %d)", __func__, geom.key.row, geom.key.col);
        }
        const size_t nPixels = width * height;
        std::vector<int32_t> labels(nPixels, -1);
        std::vector<float> probs(nPixels, 0.0f);

        if (usesRasterResolutionOverride()) {
            std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
            loadTileToMap(geom.key, pixelMap, nullptr, &in);
            for (const auto& kv : pixelMap) {
                if (kv.second.ks.empty() || kv.second.ps.empty()) {
                    continue;
                }
                const int32_t xpix = kv.first.first;
                const int32_t ypix = kv.first.second;
                if (xpix < geom.pixX0 || xpix >= geom.pixX1 || ypix < geom.pixY0 || ypix >= geom.pixY1) {
                    ++out.nOutOfRangeIgnored;
                    continue;
                }
                const size_t x0 = static_cast<size_t>(xpix - geom.pixX0);
                const size_t y0 = static_cast<size_t>(ypix - geom.pixY0);
                const size_t idx = y0 * width + x0;
                if (labels[idx] != -1) {
                    ++out.nCollisionIgnored;
                    continue;
                }
                labels[idx] = kv.second.ks[0];
                probs[idx] = kv.second.ps[0];
            }
        } else {
            in.clear();
            in.seekg(static_cast<std::streamoff>(blk.idx.st));
            if (!in.good()) {
                error("%s: Failed seeking input stream to tile %lu", __func__, bi);
            }
            TopProbs rec;
            int32_t xpix = 0;
            int32_t ypix = 0;
            uint64_t pos = blk.idx.st;
            while (readNextRecord2DAsPixel(in, pos, blk.idx.ed, xpix, ypix, rec)) {
                if (rec.ks.empty() || rec.ps.empty()) {
                    continue;
                }
                if (xpix < geom.pixX0 || xpix >= geom.pixX1 || ypix < geom.pixY0 || ypix >= geom.pixY1) {
                    ++out.nOutOfRangeIgnored;
                    continue;
                }
                const size_t x0 = static_cast<size_t>(xpix - geom.pixX0);
                const size_t y0 = static_cast<size_t>(ypix - geom.pixY0);
                const size_t idx = y0 * width + x0;
                if (labels[idx] != -1) {
                    ++out.nCollisionIgnored;
                    continue;
                }
                labels[idx] = rec.ks[0];
                probs[idx] = rec.ps[0];
            }
        }

        island_smoothing::Options smoothOpts;
        smoothOpts.fillEmpty = fillEmptyIslands;
        smoothOpts.updateProbFromWinnerMin = true;
        island_smoothing::Result ret = island_smoothing::smoothLabels8Neighborhood(
            labels, &probs, width, height, islandSmoothRounds, smoothOpts);
        (void)ret;

        size_t nOutLocal = 0;
        for (size_t i = 0; i < nPixels; ++i) {
            if (labels[i] >= 0) {
                ++nOutLocal;
            }
        }
        if (nOutLocal > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            error("%s: Too many output records in tile (%d, %d)", __func__, geom.key.row, geom.key.col);
        }
        out.nOut = static_cast<uint32_t>(nOutLocal);
        out.data.resize(nOutLocal * recSize);
        size_t off = 0;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t idx = y * width + x;
                const int32_t label = labels[idx];
                if (label < 0) {
                    continue;
                }
                const int32_t outX = static_cast<int32_t>(x) + geom.pixX0;
                const int32_t outY = static_cast<int32_t>(y) + geom.pixY0;
                const float outP = probs[idx];
                std::memcpy(out.data.data() + off, &outX, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t), &outY, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t) * 2, &label, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t) * 3, &outP, sizeof(float));
                off += recSize;
            }
        }
    };

    size_t currentOffset = 0;
    const size_t nTiles = blocks_.size();
    size_t nProcessed = 0;
    std::unique_ptr<tbb::global_control> globalLimit;
    if (useParallel) {
        globalLimit = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
    }
    for (size_t chunkBegin = 0; chunkBegin < nTiles; chunkBegin += chunkTileCount) {
        const size_t chunkEnd = std::min(nTiles, chunkBegin + chunkTileCount);
        std::vector<SmoothTileResult> chunkResults(chunkEnd - chunkBegin);
        if (useParallel && (chunkEnd - chunkBegin) > 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(chunkBegin, chunkEnd),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::ifstream in;
                    if (mode_ & 0x1) {
                        in.open(dataFile_, std::ios::binary);
                    } else {
                        in.open(dataFile_);
                    }
                    if (!in.is_open()) {
                        error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                    }
                    for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                        processTile(bi, in, chunkResults[bi - chunkBegin]);
                    }
                });
        } else {
            std::ifstream in;
            if (mode_ & 0x1) {
                in.open(dataFile_, std::ios::binary);
            } else {
                in.open(dataFile_);
            }
            if (!in.is_open()) {
                error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
            }
            for (size_t bi = chunkBegin; bi < chunkEnd; ++bi) {
                processTile(bi, in, chunkResults[bi - chunkBegin]);
            }
        }

        for (size_t bi = chunkBegin; bi < chunkEnd; ++bi) {
            const auto& res = chunkResults[bi - chunkBegin];
            const TileInfo& blk = blocks_[bi];
            IndexEntryF outEntry = blk.idx;
            outEntry.st = currentOffset;
            outEntry.n = res.nOut;
            if (!res.data.empty()) {
                if (!write_all(fdMain, res.data.data(), res.data.size())) {
                    close(fdMain);
                    close(fdIndex);
                    error("%s: Failed writing output data", __func__);
                }
            }
            outEntry.ed = currentOffset + res.data.size();
            currentOffset = outEntry.ed;
            if (!write_all(fdIndex, &outEntry, sizeof(outEntry))) {
                close(fdMain);
                close(fdIndex);
                error("%s: Failed writing output index entry", __func__);
            }
            if (res.nCollisionIgnored > 0) {
                warning("%s: Ignored %lu colliding records in tile (%d, %d)",
                    __func__, res.nCollisionIgnored, res.row, res.col);
            }
            if (res.nOutOfRangeIgnored > 0) {
                warning("%s: Ignored %lu out-of-tile records in tile (%d, %d)",
                    __func__, res.nOutOfRangeIgnored, res.row, res.col);
            }
            ++nProcessed;
            if (nProcessed % 10 == 0) {
                notice("%s: Processing tile %lu/%lu", __func__, nProcessed, nTiles);
            }
        }
    }

    close(fdMain);
    close(fdIndex);
    notice("%s: Wrote smoothed output to %s (index: %s)", __func__, outFile.c_str(), outIndex.c_str());
}

void TileOperator::spatialMetricsBasic(const std::string& outPrefix) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    auto processTile = [&](const TileInfo& blk,
            std::ifstream& in, SpatialMetricsAccum& out,
            uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) {
        DenseTile dense;
        loadDenseTile(blk, in, dense, BG, nOutOfRangeIgnored, nBadLabelIgnored);
        const size_t width = dense.geom.W;
        const size_t height = dense.geom.H;
        const size_t nPixels = width * height;
        const std::vector<uint8_t>& labels = dense.lab;

        for (size_t i = 0; i < nPixels; ++i) {
            uint8_t a = labels[i];
            if (a < BG) out.area[a]++;
        }

        if (width > 1) {
            for (size_t y = 0; y < height; ++y) {
                const size_t row = y * width;
                for (size_t x = 0; x + 1 < width; ++x) {
                    spatialAccumulateEdge(out, labels[row + x], labels[row + x + 1]);
                }
            }
        }
        if (height > 1) {
            for (size_t y = 0; y + 1 < height; ++y) {
                const size_t row = y * width;
                const size_t rowDown = row + width;
                for (size_t x = 0; x < width; ++x) {
                    spatialAccumulateEdge(out, labels[row + x], labels[rowDown + x]);
                }
            }
        }
    };

    SpatialMetricsAccum total(K);
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;
    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<SpatialMetricsAccum> tls([&] { return SpatialMetricsAccum(K); });
        tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
            [] { return std::make_pair(0ULL, 0ULL); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in;
                if (mode_ & 0x1) {
                    in.open(dataFile_, std::ios::binary);
                } else {
                    in.open(dataFile_);
                }
                if (!in.is_open()) {
                    error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                }
                auto& local = tls.local();
                auto& localIgnored = tlsIgnored.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(blocks_[bi], in, local, localIgnored.first, localIgnored.second);
                }
            });
        tls.combine_each([&](const SpatialMetricsAccum& local) {
            total.add(local);
        });
        tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
            totalOutOfRangeIgnored += localIgnored.first;
            totalBadLabelIgnored += localIgnored.second;
        });
    } else {
        std::ifstream in;
        if (mode_ & 0x1) {
            in.open(dataFile_, std::ios::binary);
        } else {
            in.open(dataFile_);
        }
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        for (const auto& blk : blocks_) {
            processTile(blk, in, total, totalOutOfRangeIgnored, totalBadLabelIgnored);
        }
    }

    std::string outSingle = outPrefix + ".stats.single.tsv";
    FILE* fpSingle = fopen(outSingle.c_str(), "w");
    if (!fpSingle) {
        error("%s: Cannot open output file %s", __func__, outSingle.c_str());
    }
    fprintf(fpSingle, "#k\tarea\tperim\n");
    for (int32_t k = 0; k <= K; ++k) {
        fprintf(fpSingle, "%d\t%llu\t%llu\n",
            k,
            static_cast<unsigned long long>(total.area[k]),
            static_cast<unsigned long long>(total.perim[k]));
    }
    fclose(fpSingle);
    notice("%s: Wrote single-channel metrics to %s", __func__, outSingle.c_str());

    std::string outPairwise = outPrefix + ".stats.pairwise.tsv";
    FILE* fpPairwise = fopen(outPairwise.c_str(), "w");
    if (!fpPairwise) {
        error("%s: Cannot open output file %s", __func__, outPairwise.c_str());
    }
    fprintf(fpPairwise, "#k\tl\tcontact\tfrac0\tfrac1\tfrac2\tdensity\n");
    for (int32_t k = 0; k < K; ++k) {
        double ak = std::max(1., static_cast<double>(total.area[k]));
        double pk = std::max(1., static_cast<double>(total.perim[k]));
        for (int32_t l = k + 1; l <= K; ++l) {
            uint64_t pkl_int = total.shared_ij[total.triIndex(k, l)];
            if (pkl_int == 0) {continue;}
            double al = std::max(1., static_cast<double>(total.area[l]));
            double pl = std::max(1., static_cast<double>(total.perim[l]));
            double pkl = static_cast<double>(pkl_int);
            fprintf(fpPairwise, "%d\t%d\t%llu\t%.2e\t%.2e\t%.2e\t%.2e\n",
                k, l, static_cast<unsigned long long>(pkl_int),
                pkl / (pk + pl - pkl), pkl / pk, pkl / pl, pkl / (ak + al));
        }
    }
    fclose(fpPairwise);
    notice("%s: Wrote pairwise metrics to %s", __func__, outPairwise.c_str());
    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
}

void TileOperator::connectedComponents(const std::string& outPrefix, uint32_t minSize,
    bool writeGeoJson) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const size_t NO_CC = std::numeric_limits<size_t>::max();
    const auto tStage1Start = std::chrono::steady_clock::now();

    std::vector<TileCCL> perTile(blocks_.size());
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;
    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(
            tbb::global_control::max_allowed_parallelism,
            static_cast<size_t>(threads_));
        tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
            [] { return std::make_pair(0ULL, 0ULL); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in;
                if (mode_ & 0x1) {
                    in.open(dataFile_, std::ios::binary);
                } else {
                    in.open(dataFile_);
                }
                if (!in.is_open()) {
                    error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                }
                auto& localIgnored = tlsIgnored.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    DenseTile dense;
                    loadDenseTile(blocks_[bi], in, dense, BG,
                        localIgnored.first, localIgnored.second);
                    perTile[bi] = tileLocalCCL(dense, BG, nullptr, writeGeoJson);
                }
            });
        tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
            totalOutOfRangeIgnored += localIgnored.first;
            totalBadLabelIgnored += localIgnored.second;
        });
    } else {
        std::ifstream in;
        if (mode_ & 0x1) {
            in.open(dataFile_, std::ios::binary);
        } else {
            in.open(dataFile_);
        }
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            DenseTile dense;
            loadDenseTile(blocks_[bi], in, dense, BG,
                totalOutOfRangeIgnored, totalBadLabelIgnored);
            perTile[bi] = tileLocalCCL(dense, BG, nullptr, writeGeoJson);
        }
    }
    const auto tStage1End = std::chrono::steady_clock::now();

    struct CCOutRec {
        uint64_t size = 0;
        uint64_t sumX = 0;
        uint64_t sumY = 0;
        PixBox box;
        bool fromBorder = false;
        size_t tileIdx = NO_CC;
        uint32_t localCid = INVALID;
        size_t borderRoot = NO_CC;
    };
    std::vector<std::vector<CCOutRec>> perLabelComponents(static_cast<size_t>(K));
    std::vector<std::unordered_map<uint64_t, uint64_t>> perLabelHist(static_cast<size_t>(K));
    std::vector<std::vector<size_t>> localOldToCcIdx;
    std::vector<BorderRemapInfo> remapByTile;
    if (writeGeoJson) {
        localOldToCcIdx.resize(blocks_.size());
        remapByTile.resize(blocks_.size());
    }
    auto addFinalComponent = [&](uint8_t lbl, uint64_t size,
                                 uint64_t sumX, uint64_t sumY,
                                 const PixBox& box,
                                 bool fromBorder,
                                 size_t tileIdx,
                                 uint32_t localCid,
                                 size_t borderRoot) {
        if (lbl >= BG || size == 0) {
            return;
        }
        perLabelHist[static_cast<size_t>(lbl)][size] += 1;
        if (size >= static_cast<uint64_t>(minSize)) {
            perLabelComponents[static_cast<size_t>(lbl)].push_back(
                CCOutRec{size, sumX, sumY, box, fromBorder, tileIdx, localCid, borderRoot});
        }
    };

    const auto tStage15Start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < perTile.size(); ++i) {
        auto& t = perTile[i];
        if (t.ncomp == 0) {
            continue;
        }
        BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);
        const BorderRemapInfo* infoPtr = &remapInfo;
        if (writeGeoJson) {
            localOldToCcIdx[i].assign(remapInfo.remap.size(), NO_CC);
            remapByTile[i] = std::move(remapInfo);
            infoPtr = &remapByTile[i];
        }
        const BorderRemapInfo& info = *infoPtr;
        for (uint32_t cid = 0; cid < info.remap.size(); ++cid) {
            if (info.remap[cid] != INVALID) {
                continue;
            }
            addFinalComponent(
                info.oldCompLabel[cid],
                static_cast<uint64_t>(info.oldCompSize[cid]),
                info.oldCompSumX[cid],
                info.oldCompSumY[cid],
                info.oldCompBox[cid],
                false,
                i,
                cid,
                NO_CC);
        }
    }
    const auto tStage15End = std::chrono::steady_clock::now();

    const auto tStage2Start = std::chrono::steady_clock::now();
    const BorderDSUState dsuState = mergeBorderComponentsWithDSU(perTile, BG, INVALID);
    std::vector<size_t> rootToCcIdx;
    if (writeGeoJson) {
        rootToCcIdx.assign(dsuState.rootSize.size(), NO_CC);
    }
    for (size_t root = 0; root < dsuState.rootSize.size(); ++root) {
        if (dsuState.rootSize[root] == 0 || dsuState.rootLabel[root] >= BG) {
            continue;
        }
        addFinalComponent(
            dsuState.rootLabel[root],
            dsuState.rootSize[root],
            dsuState.rootSumX[root],
            dsuState.rootSumY[root],
            dsuState.rootBox[root],
            true,
            NO_CC,
            INVALID,
            root);
    }
    const auto tStage2End = std::chrono::steady_clock::now();
    notice("%s: timing(ms): stage1=%lld stage1.5=%lld stage2=%lld",
        __func__,
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(tStage1End - tStage1Start).count()),
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(tStage15End - tStage15Start).count()),
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(tStage2End - tStage2Start).count()));

    std::string outCc = outPrefix + ".connected_components.tsv";
    FILE* fpCc = fopen(outCc.c_str(), "w");
    if (!fpCc) {
        error("%s: Cannot open output file %s", __func__, outCc.c_str());
    }
    fprintf(fpCc, "#k\tcc_idx\tsize\tcentroid_x\tcentroid_y\txmin\txmax\tymin\tymax\n");
    for (int32_t k = 0; k < K; ++k) {
        auto& vec = perLabelComponents[static_cast<size_t>(k)];
        std::sort(vec.begin(), vec.end(),
            [](const CCOutRec& a, const CCOutRec& b) {
                if (a.size != b.size) return a.size > b.size;
                if (a.box.minX != b.box.minX) return a.box.minX < b.box.minX;
                if (a.box.minY != b.box.minY) return a.box.minY < b.box.minY;
                if (a.box.maxX != b.box.maxX) return a.box.maxX < b.box.maxX;
                return a.box.maxY < b.box.maxY;
            });
        for (size_t i = 0; i < vec.size(); ++i) {
            const double cx = static_cast<double>(vec[i].sumX) / static_cast<double>(vec[i].size);
            const double cy = static_cast<double>(vec[i].sumY) / static_cast<double>(vec[i].size);
            fprintf(fpCc, "%d\t%zu\t%llu\t%.6f\t%.6f\t%d\t%d\t%d\t%d\n",
                k,
                i,
                static_cast<unsigned long long>(vec[i].size),
                cx,
                cy,
                vec[i].box.minX,
                vec[i].box.maxX,
                vec[i].box.minY,
                vec[i].box.maxY);
            if (!writeGeoJson) {
                continue;
            }
            if (vec[i].fromBorder) {
                if (vec[i].borderRoot >= rootToCcIdx.size()) {
                    error("%s: Border root index out of range", __func__);
                }
                rootToCcIdx[vec[i].borderRoot] = i;
            } else {
                if (vec[i].tileIdx >= localOldToCcIdx.size()
                    || vec[i].localCid >= localOldToCcIdx[vec[i].tileIdx].size()) {
                    error("%s: Tile-local component index out of range", __func__);
                }
                localOldToCcIdx[vec[i].tileIdx][vec[i].localCid] = i;
            }
        }
    }
    fclose(fpCc);
    notice("%s: Wrote connected components to %s", __func__, outCc.c_str());

    std::string outHist = outPrefix + ".connected_components_hist.tsv";
    FILE* fpHist = fopen(outHist.c_str(), "w");
    if (!fpHist) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fpHist, "#k\tsize\tn_components\n");
    for (int32_t k = 0; k < K; ++k) {
        const auto& hist = perLabelHist[static_cast<size_t>(k)];
        for (const auto& kv : hist) {
            fprintf(fpHist, "%d\t%llu\t%llu\n",
                k,
                static_cast<unsigned long long>(kv.first),
                static_cast<unsigned long long>(kv.second));
        }
    }
    fclose(fpHist);
    notice("%s: Wrote connected component size histogram to %s", __func__, outHist.c_str());

    if (writeGeoJson) {
        std::vector<std::vector<Clipper2Lib::Paths64>> componentPaths(static_cast<size_t>(K));
        for (int32_t k = 0; k < K; ++k) {
            componentPaths[static_cast<size_t>(k)].resize(
                perLabelComponents[static_cast<size_t>(k)].size());
        }

        for (size_t bi = 0; bi < perTile.size(); ++bi) {
            const auto& info = remapByTile[bi];
            if (info.remap.empty()) {
                continue;
            }
            if (perTile[bi].pixelCid.empty()) {
                error("%s: Missing pixel-to-component map for GeoJSON export", __func__);
            }
            TileGeom geom;
            initTileGeom(blocks_[bi], geom);
            for (uint32_t oldCid = 0; oldCid < info.remap.size(); ++oldCid) {
                const uint8_t lbl = info.oldCompLabel[oldCid];
                if (lbl >= BG) {
                    continue;
                }
                size_t ccIdx = NO_CC;
                const uint32_t newCid = info.remap[oldCid];
                if (newCid == INVALID) {
                    if (oldCid >= localOldToCcIdx[bi].size()) {
                        error("%s: Missing non-border component lookup", __func__);
                    }
                    ccIdx = localOldToCcIdx[bi][oldCid];
                } else {
                    if (bi >= dsuState.tileRoot.size() || newCid >= dsuState.tileRoot[bi].size()) {
                        error("%s: Missing border root lookup", __func__);
                    }
                    const size_t root = dsuState.tileRoot[bi][newCid];
                    if (root < rootToCcIdx.size()) {
                        ccIdx = rootToCcIdx[root];
                    }
                }
                if (ccIdx == NO_CC) {
                    continue;
                }
                if (ccIdx >= componentPaths[static_cast<size_t>(lbl)].size()) {
                    error("%s: GeoJSON component rank out of range", __func__);
                }
                Clipper2Lib::Paths64 paths = buildLabelComponentPolygons(
                    perTile[bi].pixelCid,
                    geom.W,
                    oldCid,
                    info.oldCompBox[oldCid],
                    geom,
                    0,
                    0.0);
                auto& dst = componentPaths[static_cast<size_t>(lbl)][ccIdx];
                dst.insert(dst.end(), paths.begin(), paths.end());
            }
        }

        const double coordScale = static_cast<double>(
            getRasterPixelResolution() > 0.0f ? getRasterPixelResolution() : 1.0f);
        const double scaleInv = 1.0 / coordScale;
        for (int32_t k = 0; k < K; ++k) {
            nlohmann::json features = nlohmann::json::array();
            const auto& vec = perLabelComponents[static_cast<size_t>(k)];
            for (size_t i = 0; i < vec.size(); ++i) {
                Clipper2Lib::Paths64 merged = cleanupMaskPolygons(componentPaths[static_cast<size_t>(k)][i], 0, 0.0);
                nlohmann::json geometry = maskPathsToMultiPolygonGeoJSON(merged);
                if (scaleInv != 1.0) {
                    rescaleGeometryCoordinates(geometry["coordinates"], scaleInv);
                }
                if (geometry["type"] == "MultiPolygon"
                    && geometry["coordinates"].is_array()
                    && geometry["coordinates"].size() == 1) {
                    geometry["type"] = "Polygon";
                    geometry["coordinates"] = geometry["coordinates"][0];
                }
                const double cx = static_cast<double>(vec[i].sumX) / static_cast<double>(vec[i].size);
                const double cy = static_cast<double>(vec[i].sumY) / static_cast<double>(vec[i].size);
                nlohmann::json feature;
                feature["type"] = "Feature";
                feature["properties"] = {
                    {"k", k},
                    {"cc_idx", i},
                    {"size", vec[i].size},
                    {"centroid_x", cx},
                    {"centroid_y", cy},
                    {"xmin", vec[i].box.minX},
                    {"xmax", vec[i].box.maxX},
                    {"ymin", vec[i].box.minY},
                    {"ymax", vec[i].box.maxY}
                };
                feature["geometry"] = std::move(geometry);
                features.push_back(std::move(feature));
            }

            nlohmann::json featureCollection;
            featureCollection["type"] = "FeatureCollection";
            featureCollection["features"] = std::move(features);
            const std::string outGeo = outPrefix + ".connected_components.k" + std::to_string(k) + ".geojson";
            std::ofstream geoOut(outGeo);
            if (!geoOut.is_open()) {
                error("%s: Cannot open output file %s", __func__, outGeo.c_str());
            }
            geoOut << featureCollection.dump();
            geoOut.close();
            notice("%s: Wrote per-label connected component polygons to %s",
                __func__, outGeo.c_str());
        }
    }

    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
}

void TileOperator::profileShellAndSurface(const std::string& outPrefix,
    const std::vector<int32_t>& radii, int32_t dMax,
    uint32_t minComponentSize, uint32_t minPixPerTilePerLabel) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    if (dMax < 0) {
        error("%s: dMax must be >= 0", __func__);
    }
    std::vector<int32_t> radiiSorted;
    radiiSorted.reserve(radii.size());
    for (int32_t r : radii) {
        if (r < 0) continue;
        radiiSorted.push_back(r);
    }
    if (radiiSorted.empty()) {
        error("%s: At least one non-negative radius is required", __func__);
    }
    std::sort(radiiSorted.begin(), radiiSorted.end());
    radiiSorted.erase(std::unique(radiiSorted.begin(), radiiSorted.end()), radiiSorted.end());
    const int32_t rMax = radiiSorted.back();
    const int32_t D = std::max(rMax, dMax + 1);
    if (D > 65534) {
        error("%s: Distance cap too large (%d), must be <= 65534", __func__, D);
    }

    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();

    std::vector<DenseTile> tiles(blocks_.size());
    std::vector<uint64_t> areaTot(static_cast<size_t>(K + 1), 0);
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;
    std::vector<std::vector<PixelRef>> seedsBoundaryBig(static_cast<size_t>(K));
    {
        const bool needTileLabelCount = (minPixPerTilePerLabel > 0);
        std::vector<std::vector<uint32_t>> tileLabelCount;
        if (needTileLabelCount) {
            tileLabelCount.assign(
                blocks_.size(), std::vector<uint32_t>(static_cast<size_t>(K), 0));
        }

        if (threads_ > 1 && blocks_.size() > 1) {
            tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
            tbb::combinable<std::vector<uint64_t>> tlsArea([&] {
                return std::vector<uint64_t>(static_cast<size_t>(K + 1), 0);
            });
            tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
                [] { return std::make_pair(0ULL, 0ULL); });
            tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::ifstream in;
                    if (mode_ & 0x1) {
                        in.open(dataFile_, std::ios::binary);
                    } else {
                        in.open(dataFile_);
                    }
                    if (!in.is_open()) {
                        error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                    }
                    auto& localArea = tlsArea.local();
                    auto& localIgnored = tlsIgnored.local();
                    for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                        loadDenseTile(blocks_[bi], in, tiles[bi], BG, localIgnored.first, localIgnored.second);
                        if (tiles[bi].lab.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                            error("%s: Tile too large for local indexing", __func__);
                        }
                        for (size_t li = 0; li < tiles[bi].lab.size(); ++li) {
                            const uint8_t lbl = tiles[bi].lab[li];
                            localArea[lbl] += 1;
                            if (needTileLabelCount && lbl < BG) {
                                tileLabelCount[bi][static_cast<size_t>(lbl)] += 1;
                            }
                        }
                    }
                });
            tlsArea.combine_each([&](const std::vector<uint64_t>& localArea) {
                for (size_t i = 0; i < areaTot.size(); ++i) areaTot[i] += localArea[i];
            });
            tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
                totalOutOfRangeIgnored += localIgnored.first;
                totalBadLabelIgnored += localIgnored.second;
            });
        } else {
            std::ifstream in;
            if (mode_ & 0x1) {
                in.open(dataFile_, std::ios::binary);
            } else {
                in.open(dataFile_);
            }
            if (!in.is_open()) {
                error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
            }
            for (size_t bi = 0; bi < blocks_.size(); ++bi) {
                loadDenseTile(blocks_[bi], in, tiles[bi], BG, totalOutOfRangeIgnored, totalBadLabelIgnored);
                if (tiles[bi].lab.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    error("%s: Tile too large for local indexing", __func__);
                }
                for (size_t li = 0; li < tiles[bi].lab.size(); ++li) {
                    const uint8_t lbl = tiles[bi].lab[li];
                    areaTot[lbl] += 1;
                    if (needTileLabelCount && lbl < BG) {
                        tileLabelCount[bi][static_cast<size_t>(lbl)] += 1;
                    }
                }
            }
        }

        computeTileBoundaryMasks(tiles);
        std::vector<TileCCL> perTile(tiles.size());
        if (threads_ > 1 && tiles.size() > 1) {
            tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
            tbb::parallel_for(tbb::blocked_range<size_t>(0, tiles.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                        perTile[bi] = tileLocalCCL(tiles[bi], BG, &tiles[bi].boundary, false);
                    }
                });
        } else {
            for (size_t bi = 0; bi < tiles.size(); ++bi) {
                perTile[bi] = tileLocalCCL(tiles[bi], BG, &tiles[bi].boundary, false);
            }
        }

        std::vector<std::vector<uint8_t>> tileCompBig(tiles.size());
        for (size_t i = 0; i < perTile.size(); ++i) {
            auto& t = perTile[i];
            if (t.ncomp == 0) continue;
            const BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);

            std::vector<uint32_t> bndPixKeep;
            std::vector<uint32_t> bndCidKeep;
            bndPixKeep.reserve(t.bndPix.size());
            bndCidKeep.reserve(t.bndCid.size());
            for (size_t bi = 0; bi < t.bndPix.size(); ++bi) {
                const uint32_t oldCid = t.bndCid[bi];
                if (oldCid >= remapInfo.remap.size()) {
                    error("%s: Boundary cid out of range", __func__);
                }
                const uint32_t pix = t.bndPix[bi];
                const uint8_t lbl = tiles[i].lab[pix];
                if (lbl >= BG) continue;
                if (remapInfo.remap[oldCid] == INVALID) {
                    if (remapInfo.oldCompSize[oldCid] < minComponentSize) continue;
                    if (needTileLabelCount &&
                        tileLabelCount[i][static_cast<size_t>(lbl)] < minPixPerTilePerLabel) {
                        continue;
                    }
                    seedsBoundaryBig[static_cast<size_t>(lbl)].push_back(
                        PixelRef{static_cast<uint32_t>(i), pix});
                } else {
                    bndPixKeep.push_back(pix);
                    bndCidKeep.push_back(remapInfo.remap[oldCid]);
                }
            }
            t.bndPix.swap(bndPixKeep);
            t.bndCid.swap(bndCidKeep);
            tileCompBig[i].assign(static_cast<size_t>(t.ncomp), 0);
        }

        const BorderDSUState dsuState = mergeBorderComponentsWithDSU(perTile, BG, INVALID);
        for (size_t i = 0; i < perTile.size(); ++i) {
            auto& t = perTile[i];
            auto& big = tileCompBig[i];
            if (big.size() != static_cast<size_t>(t.ncomp)) {
                big.assign(static_cast<size_t>(t.ncomp), 0);
            }
            const auto& tileRoot = dsuState.tileRoot[i];
            for (size_t cid = 0; cid < t.ncomp; ++cid) {
                if (cid >= tileRoot.size()) {
                    error("%s: Missing root id for border component", __func__);
                }
                const size_t root = tileRoot[cid];
                big[cid] = (root < dsuState.rootSize.size() && dsuState.rootSize[root] >= minComponentSize) ? 1 : 0;
            }
            for (size_t bi = 0; bi < t.bndPix.size(); ++bi) {
                const uint32_t cid = t.bndCid[bi];
                if (cid >= big.size() || !big[cid]) continue;
                const uint32_t pix = t.bndPix[bi];
                const uint8_t lbl = tiles[i].lab[pix];
                if (lbl >= BG) continue;
                if (needTileLabelCount &&
                    tileLabelCount[i][static_cast<size_t>(lbl)] < minPixPerTilePerLabel) {
                    continue;
                }
                seedsBoundaryBig[static_cast<size_t>(lbl)].push_back(
                    PixelRef{static_cast<uint32_t>(i), pix});
            }
        }
    }

    struct TileNeighbor {
        int32_t left = -1;
        int32_t right = -1;
        int32_t up = -1;
        int32_t down = -1;
    };
    std::vector<TileNeighbor> nbr(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileKey key = tiles[i].geom.key;
        auto it = tile_lookup_.find(TileKey{key.row, key.col - 1});
        if (it != tile_lookup_.end()) nbr[i].left = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (it != tile_lookup_.end()) nbr[i].right = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row - 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].up = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].down = static_cast<int32_t>(it->second);
    }

    auto seamNeighborIndex = [&](uint32_t tileIdx, int32_t neighborIdx,
        int32_t ngx, int32_t ngy) -> uint32_t {
        if (neighborIdx < 0) {
            return INVALID;
        }
        const DenseTile& nt = tiles[static_cast<size_t>(neighborIdx)];
        if (ngx < nt.geom.pixX0 || ngx >= nt.geom.pixX1 ||
            ngy < nt.geom.pixY0 || ngy >= nt.geom.pixY1) {
            return INVALID;
        }
        const size_t nx = static_cast<size_t>(ngx - nt.geom.pixX0);
        const size_t ny = static_cast<size_t>(ngy - nt.geom.pixY0);
        (void)tileIdx;
        return static_cast<uint32_t>(ny * nt.geom.W + nx);
    };

    const uint16_t INF = static_cast<uint16_t>(D + 1);
    struct TileDistBuf {
        std::vector<uint16_t> dist;
        std::vector<uint32_t> touched;
    };
    std::vector<TileDistBuf> distBuf(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        distBuf[i].dist.assign(tiles[i].lab.size(), INF);
        distBuf[i].touched.reserve(std::min<size_t>(tiles[i].lab.size(), 4096));
    }
    std::vector<uint8_t> tileActiveMark(tiles.size(), 0);
    std::vector<uint32_t> activeTiles;
    activeTiles.reserve(std::min<size_t>(tiles.size(), 1024));

    std::string outShell = outPrefix + ".shell.tsv";
    FILE* fpShell = fopen(outShell.c_str(), "w");
    if (!fpShell) {
        error("%s: Cannot open output file %s", __func__, outShell.c_str());
    }
    fprintf(fpShell, "#K_focal\tK2\tr\tn_within\tn_K2_total\n");

    std::string outSurface = outPrefix + ".surface.tsv";
    FILE* fpSurface = fopen(outSurface.c_str(), "w");
    if (!fpSurface) {
        fclose(fpShell);
        error("%s: Cannot open output file %s", __func__, outSurface.c_str());
    }
    fprintf(fpSurface, "#from_K1\tto_K2\td\tcount\n");

    auto shellIndex = [&](int32_t b, int32_t d) -> size_t {
        return static_cast<size_t>(b) * static_cast<size_t>(rMax + 1) + static_cast<size_t>(d);
    };
    auto surfIndex = [&](int32_t b, int32_t d) -> size_t {
        return static_cast<size_t>(b) * static_cast<size_t>(dMax + 1) + static_cast<size_t>(d);
    };

    std::vector<PixelRef> frontier;
    std::vector<PixelRef> nextFrontier;
    for (int32_t A = 0; A < K; ++A) {
        for (uint32_t ti : activeTiles) {
            auto& db = distBuf[ti];
            if (db.touched.size() > db.dist.size() / 2) {
                std::fill(db.dist.begin(), db.dist.end(), INF);
            } else {
                for (uint32_t idx : db.touched) db.dist[idx] = INF;
            }
            tileActiveMark[ti] = 0;
        }
        activeTiles.clear();

        frontier.clear();
        nextFrontier.clear();
        const auto& seedsA = seedsBoundaryBig[static_cast<size_t>(A)];
        frontier.reserve(seedsA.size());
        for (const auto& ref : seedsA) {
            auto& db = distBuf[ref.tileIdx];
            if (db.dist[ref.localIdx] == 0) continue;
            db.dist[ref.localIdx] = 0;
            db.touched.push_back(ref.localIdx);
            if (!tileActiveMark[ref.tileIdx]) {
                tileActiveMark[ref.tileIdx] = 1;
                activeTiles.push_back(ref.tileIdx);
            }
            frontier.push_back(ref);
        }
        if (frontier.empty()) continue;

        std::vector<uint64_t> shellCount(static_cast<size_t>(K + 1) * static_cast<size_t>(rMax + 1), 0);
        std::vector<uint64_t> surfHist(static_cast<size_t>(K + 1) * static_cast<size_t>(dMax + 1), 0);

        for (int32_t d = 0; d <= D && !frontier.empty(); ++d) {
            nextFrontier.clear();
            auto tryPush = [&](uint32_t nti, uint32_t nli, uint16_t nd) {
                if (tiles[nti].lab[nli] == static_cast<uint8_t>(A)) return;
                auto& db = distBuf[nti];
                if (db.dist[nli] != INF) return;
                db.dist[nli] = nd;
                db.touched.push_back(nli);
                if (!tileActiveMark[nti]) {
                    tileActiveMark[nti] = 1;
                    activeTiles.push_back(nti);
                }
                nextFrontier.push_back(PixelRef{nti, nli});
            };

            for (const auto& ref : frontier) {
                const DenseTile& tile = tiles[ref.tileIdx];
                const size_t li = ref.localIdx;
                const uint8_t B = tile.lab[li];
                if (B != static_cast<uint8_t>(A)) {
                    if (d <= rMax) {
                        shellCount[shellIndex(B, d)] += 1;
                    }
                    if (tile.boundary[li]) {
                        const int32_t dReport = (d > 0) ? (d - 1) : 0;
                        if (dReport <= dMax) {
                            surfHist[surfIndex(B, dReport)] += 1;
                        }
                    }
                }
                if (d == D) continue;

                const size_t x = li % tile.geom.W;
                const size_t y = li / tile.geom.W;
                const uint16_t nd = static_cast<uint16_t>(d + 1);
                if (x > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - 1), nd);
                } else if (nbr[ref.tileIdx].left >= 0) {
                    const int32_t ngx = tile.geom.pixX0 - 1;
                    const int32_t ngy = tile.geom.pixY0 + static_cast<int32_t>(y);
                    const uint32_t nli = seamNeighborIndex(ref.tileIdx, nbr[ref.tileIdx].left, ngx, ngy);
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].left), nli, nd);
                    }
                }
                if (x + 1 < tile.geom.W) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + 1), nd);
                } else if (nbr[ref.tileIdx].right >= 0) {
                    const int32_t ngx = tile.geom.pixX1;
                    const int32_t ngy = tile.geom.pixY0 + static_cast<int32_t>(y);
                    const uint32_t nli = seamNeighborIndex(ref.tileIdx, nbr[ref.tileIdx].right, ngx, ngy);
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].right), nli, nd);
                    }
                }
                if (y > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - tile.geom.W), nd);
                } else if (nbr[ref.tileIdx].up >= 0) {
                    const int32_t ngx = tile.geom.pixX0 + static_cast<int32_t>(x);
                    const int32_t ngy = tile.geom.pixY0 - 1;
                    const uint32_t nli = seamNeighborIndex(ref.tileIdx, nbr[ref.tileIdx].up, ngx, ngy);
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].up), nli, nd);
                    }
                }
                if (y + 1 < tile.geom.H) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + tile.geom.W), nd);
                } else if (nbr[ref.tileIdx].down >= 0) {
                    const int32_t ngx = tile.geom.pixX0 + static_cast<int32_t>(x);
                    const int32_t ngy = tile.geom.pixY1;
                    const uint32_t nli = seamNeighborIndex(ref.tileIdx, nbr[ref.tileIdx].down, ngx, ngy);
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].down), nli, nd);
                    }
                }
            }
            frontier.swap(nextFrontier);
        }

        for (int32_t b = 0; b <= K; ++b) {
            for (int32_t d = 1; d <= rMax; ++d) {
                shellCount[shellIndex(b, d)] += shellCount[shellIndex(b, d - 1)];
            }
        }
        for (int32_t b = 0; b <= K; ++b) {
            if (b == A) continue;
            const uint64_t totalB = areaTot[static_cast<size_t>(b)];
            for (int32_t r : radiiSorted) {
                const uint64_t within = shellCount[shellIndex(b, r)];
                fprintf(fpShell, "%d\t%d\t%d\t%llu\t%llu\n",
                    A, b, r,
                    static_cast<unsigned long long>(within),
                    static_cast<unsigned long long>(totalB));
            }
        }
        for (int32_t b = 0; b <= K; ++b) {
            if (b == A) continue;
            for (int32_t d = 0; d <= dMax; ++d) {
                const uint64_t ct = surfHist[surfIndex(b, d)];
                if (ct == 0) continue;
                fprintf(fpSurface, "%d\t%d\t%d\t%llu\n",
                    b, A, d, static_cast<unsigned long long>(ct));
            }
        }
    }

    fclose(fpShell);
    fclose(fpSurface);
    notice("%s: Wrote shell occupancy to %s", __func__, outShell.c_str());
    notice("%s: Wrote surface distance histogram to %s", __func__, outSurface.c_str());
    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
}
