#ifndef PRUNED_LANDMARK_LABELING_H_
#define PRUNED_LANDMARK_LABELING_H_

#include <malloc.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>
#include <fstream>
#include <utility>
using namespace std;
// NOTE: Directed graphs are supported.

template<int kNumBitParallelRoots = 0>
class PrunedLandmarkLabeling {
 public:
  // Constructs an index from a graph, given as a list of edges.
  // Vertices should be described by numbers starting from zero.
  // Returns |true| when successful.
  bool ConstructIndex(string map_path, string centrality_path);
  // Returns distance between vertices |v| and |w| if they are connected.
  // Otherwise, returns |INT_MAX|.
  inline int QueryDistance(int v, int w);


  int GetNumVertices() { return num_v_; }
  void Free();
  std::string Statistics();

  // Store the index to a file
  bool StoreIndex(const char* filename);

  // Load the index from a file
  bool LoadIndex(const char* filename);

  // Get the number of items in inIndex[v] and outIndex[v]
  std::pair<int, int> GetNumIndexItems(int v) const {
    return {inIndex[v].num_items, outIndex[v].num_items};
  }

  PrunedLandmarkLabeling()
      : num_v_(0), inIndex(NULL), outIndex(NULL), time_load_(0), time_indexing_(0) {} //tip
  virtual ~PrunedLandmarkLabeling() {
    Free();
  }

 private:
  static const uint8_t INF8;  // For unreachable pairs
  long long search_space = 0;
  long long visited_space = 0;
  size_t spt_v_memory = 0;
  size_t spt_d_memory = 0;
  // in
  struct inLabel {
    uint8_t bpspt_d[kNumBitParallelRoots];   //？
    uint64_t bpspt_s[kNumBitParallelRoots][2];  // [0]: S^{-1}, [1]: S^{0}
    uint32_t *spt_v;
    uint8_t *spt_d;
    int num_items;
  } __attribute__((aligned(64)));  // Aligned for cache lines

  //out
  struct outLabel {
    uint8_t bpspt_d[kNumBitParallelRoots];   //？
    uint64_t bpspt_s[kNumBitParallelRoots][2];  // [0]: S^{-1}, [1]: S^{0}
    uint32_t *spt_v;
    uint8_t *spt_d;
    int num_items;
  } __attribute__((aligned(64)));  // Aligned for cache lines

 
  int num_v_;
  inLabel *inIndex;  //inIndex : INDEX
  outLabel *outIndex;  //outIndex : INDEX
  double GetCurrentTimeSec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

  // Statistics
  double time_load_, time_indexing_;
};

template<int kNumBitParallelRoots>
const uint8_t PrunedLandmarkLabeling<kNumBitParallelRoots>::INF8 = 100; // 100 = 01100100

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::ConstructIndex(string map_path, string centrality_path) {
  Free();

  time_load_ = -GetCurrentTimeSec();

  // read graph
  std::ifstream edgesFileStream(map_path);    
  
  std::vector<std::pair<int, int> > es;
  for (int v, w; edgesFileStream >> v >> w; ) {
    es.push_back(std::make_pair(v, w));
  }
  
  int E = es.size();
  int &V = num_v_;
  V = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    V = std::max(V, std::max(es[i].first, es[i].second) + 1);
  }  
  std::vector<std::vector<int> > inEdges(V);
  std::vector<std::vector<int> > outEdges(V);
  for (size_t i = 0; i < es.size(); ++i) {
    int v = es[i].first, w = es[i].second;
    outEdges[v].push_back(w);
    inEdges[w].push_back(v); //tip  
  }

  time_load_ += GetCurrentTimeSec();

  // in
  inIndex = (inLabel*)memalign(64, V * sizeof(inLabel));
  if (inIndex == NULL) {
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < V; ++v) {
    inIndex[v].spt_v = NULL;
    inIndex[v].spt_d = NULL;
    inIndex[v].num_items = 0;
  }

  //out
  outIndex = (outLabel*)memalign(64, V * sizeof(outLabel));
  if (outIndex == NULL) {
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < V; ++v) {
    outIndex[v].spt_v = NULL;
    outIndex[v].spt_d = NULL;
    outIndex[v].num_items = 0;
  }

  //
  // Order vertices by decreasing order of degree
  //
  time_indexing_ = -GetCurrentTimeSec();
  std::vector<int> inv(V);  // new label -> old label
  {
    // Order
    std::vector<std::pair<float, int> > deg;
    std::ifstream centralityFileStream(centrality_path);
    if (!centralityFileStream.is_open()) {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }
    float f;
    int i;
    while (centralityFileStream >> f >> i) {
        deg.push_back(std::make_pair(f, i));
    }
    centralityFileStream.close();

    
    for (int i = 0; i < V; ++i) inv[i] = deg[i].second;//inv: order: 1 0

    // Relabel the vertex IDs
    std::vector<int> rank(V);
    for (int i = 0; i < V; ++i) rank[deg[i].second] = i; // rank: 1 0
    std::vector<std::vector<int> > new_inEdges(V);
    std::vector<std::vector<int> > new_outEdges(V);
    for (int v = 0; v < V; ++v) {
      for (size_t i = 0; i < inEdges[v].size(); ++i) {
        new_inEdges[rank[v]].push_back(rank[inEdges[v][i]]); //new_inEdges: 存的是rank， 按centrality大小
      }
      for (size_t i = 0; i < outEdges[v].size(); ++i) {
        new_outEdges[rank[v]].push_back(rank[outEdges[v][i]]); //new_outEdges: 存的是rank， 按centrality大小
      }
    }
    inEdges.swap(new_inEdges);
    outEdges.swap(new_outEdges);
  }

  //
  // Bit-parallel labeling
  //
  std::vector<bool> in_usd(V, false);  // Used as root? (in new label)
  std::vector<bool> out_usd(V, false);
  {
    std::vector<uint8_t> tmp_d(V);
    std::vector<std::pair<uint64_t, uint64_t> > tmp_s(V);
    std::vector<int> que(V);
    std::vector<std::pair<int, int> > sibling_es(E); //E0
    std::vector<std::pair<int, int> > child_es(E);  //E1

    int r = 0;
    for (int i_bpspt = 0; i_bpspt < kNumBitParallelRoots; ++i_bpspt) {
      while (r < V && in_usd[r] && out_usd[r] ) ++r;  
      if (r == V) {
        for (int v = 0; v < V; ++v) inIndex[v].bpspt_d[i_bpspt] = INF8;
        for (int v = 0; v < V; ++v) outIndex[v].bpspt_d[i_bpspt] = INF8;
        continue;
      }
      
      //in
      in_usd[r] = true;
      {
        fill(tmp_d.begin(), tmp_d.end(), INF8);
        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

        int que_t0 = 0, que_t1 = 0, que_h = 0;
        que[que_h++] = r; //把当前root放到quene里
        tmp_d[r] = 0;   
        que_t1 = que_h;  

        int ns = 0;
        std::vector<int> vs;
        sort(inEdges[r].begin(), inEdges[r].end());

        for (size_t i = 0; i < inEdges[r].size(); ++i) {
          int v = inEdges[r][i];
          if (!in_usd[v]) {
            in_usd[v] = true;
            que[que_h++] = v;
            tmp_d[v] = 1;
            tmp_s[v].first = 1ULL << ns;  // S[-1] = 1
            vs.push_back(v);
            if (++ns == 64) break;
          }
        }

        for (int d = 0; que_t0 < que_h; ++d) {
          int num_sibling_es = 0, num_child_es = 0;

          for (int que_i = que_t0; que_i < que_t1; ++que_i) {
            int v = que[que_i];

            for (size_t i = 0; i < inEdges[v].size(); ++i) {
              int tv = inEdges[v][i];
              int td = d + 1;

              if (d > tmp_d[tv]);
              else if (d == tmp_d[tv]) {
                if (v < tv) {
                  sibling_es[num_sibling_es].first  = v;
                  sibling_es[num_sibling_es].second = tv;
                  ++num_sibling_es;
                }
              } else {
                if (tmp_d[tv] == INF8) {
                  que[que_h++] = tv;
                  tmp_d[tv] = td;
                }
                child_es[num_child_es].first  = v;
                child_es[num_child_es].second = tv;
                ++num_child_es;
              }
            }
          }

          for (int i = 0; i < num_sibling_es; ++i) {
            int v = sibling_es[i].first, w = sibling_es[i].second;
            tmp_s[v].second |= tmp_s[w].first;
            tmp_s[w].second |= tmp_s[v].first;
          }
          for (int i = 0; i < num_child_es; ++i) {
            int v = child_es[i].first, c = child_es[i].second;
            tmp_s[c].first  |= tmp_s[v].first;
            tmp_s[c].second |= tmp_s[v].second;
          }

          que_t0 = que_t1;
          que_t1 = que_h;
        }

        for (int v = 0; v < V; ++v) {
          outIndex[inv[v]].bpspt_d[i_bpspt] = tmp_d[v];
          outIndex[inv[v]].bpspt_s[i_bpspt][0] = tmp_s[v].first;
          outIndex[inv[v]].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
        }
      }

      // out
      out_usd[r] = true;
      {
        fill(tmp_d.begin(), tmp_d.end(), INF8);
        fill(tmp_s.begin(), tmp_s.end(), std::make_pair(0, 0));

        int que_t0 = 0, que_t1 = 0, que_h = 0;
        que[que_h++] = r; 
        tmp_d[r] = 0;   
        que_t1 = que_h;  

        int ns = 0;
        std::vector<int> vs;
        sort(outEdges[r].begin(), outEdges[r].end());

        for (size_t i = 0; i < outEdges[r].size(); ++i) {
          int v = outEdges[r][i];
          if (!out_usd[v]) {
            out_usd[v] = true;
            que[que_h++] = v;
            tmp_d[v] = 1;
            tmp_s[v].first = 1ULL << ns; 
            vs.push_back(v);
            if (++ns == 64) break;
          }
        }

        for (int d = 0; que_t0 < que_h; ++d) {
          int num_sibling_es = 0, num_child_es = 0;

          for (int que_i = que_t0; que_i < que_t1; ++que_i) {
            int v = que[que_i];

            for (size_t i = 0; i < outEdges[v].size(); ++i) {
              int tv = outEdges[v][i];
              int td = d + 1;

              if (d > tmp_d[tv]);
              else if (d == tmp_d[tv]) {
                if (v < tv) {
                  sibling_es[num_sibling_es].first  = v;
                  sibling_es[num_sibling_es].second = tv;
                  ++num_sibling_es;
                }
              } else {
                if (tmp_d[tv] == INF8) {
                  que[que_h++] = tv;
                  tmp_d[tv] = td;
                }
                child_es[num_child_es].first  = v;
                child_es[num_child_es].second = tv;
                ++num_child_es;
              }
            }
          }

          for (int i = 0; i < num_sibling_es; ++i) {
            int v = sibling_es[i].first, w = sibling_es[i].second;
            tmp_s[v].second |= tmp_s[w].first;
            tmp_s[w].second |= tmp_s[v].first;
          }
          for (int i = 0; i < num_child_es; ++i) {
            int v = child_es[i].first, c = child_es[i].second;
            tmp_s[c].first  |= tmp_s[v].first;
            tmp_s[c].second |= tmp_s[v].second;
          }

          que_t0 = que_t1;
          que_t1 = que_h;
        }

        for (int v = 0; v < V; ++v) {
          inIndex[inv[v]].bpspt_d[i_bpspt] = tmp_d[v];
          inIndex[inv[v]].bpspt_s[i_bpspt][0] = tmp_s[v].first;
          inIndex[inv[v]].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
        }
      }
    }
  }

  //
  // Pruned labeling
  //
  {
    // Sentinel (V, INF8) is added to all the vertices
    std::vector<std::pair<std::vector<int>, std::vector<uint8_t> > >
        tmp_in_idx(V, make_pair(std::vector<int>(1, V),
                             std::vector<uint8_t>(1, INF8)));
    std::vector<std::pair<std::vector<int>, std::vector<uint8_t> > >
        tmp_out_idx(V, make_pair(std::vector<int>(1, V),
                             std::vector<uint8_t>(1, INF8)));

    std::vector<bool> vis(V);
    std::vector<int> que(V);
    std::vector<uint8_t> dst_r(V + 1, INF8);
    for (int r = 0; r < V; ++r) {
      if (in_usd[r] && out_usd[r]) continue;
      // std::cout << r << std::endl;
      // in- backward
      {
        inLabel &idx_r = inIndex[inv[r]];
        const std::pair<std::vector<int>, std::vector<uint8_t> >
            &tmp_idx_r = tmp_in_idx[r];
        for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
          dst_r[tmp_idx_r.first[i]] = tmp_idx_r.second[i];
        }

        int que_t0 = 0, que_t1 = 0, que_h = 0;
        que[que_h++] = r;
        vis[r] = true;
        que_t1 = que_h;

        for (int d = 0; que_t0 < que_h; ++d) {
          for (int que_i = que_t0; que_i < que_t1; ++que_i) {
            int v = que[que_i];
            std::pair<std::vector<int>, std::vector<uint8_t> >
                &tmp_idx_v = tmp_out_idx[v];
            outLabel &idx_v = outIndex[inv[v]];  //

            // Prefetch
            _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
            _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v.first[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v.second[0], _MM_HINT_T0);

            // Prune?
            if (in_usd[v]) continue;
            for (int i = 0; i < kNumBitParallelRoots; ++i) {
              int td = idx_r.bpspt_d[i] + idx_v.bpspt_d[i];
              if (td - 2 <= d) {
                td +=
                    (idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
                    ((idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
                    (idx_r.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                    ? -1 : 0;
                if (td <= d) goto pruned1;
              }
            }
            // here
            visited_space += 1;
            for (size_t i = 0; i < tmp_idx_v.first.size(); ++i) {
              int w = tmp_idx_v.first[i];
              int td = tmp_idx_v.second[i] + dst_r[w];
              if (td <= d) goto pruned1;
            }

            // Traverse
            tmp_idx_v.first .back() = r;
            tmp_idx_v.second.back() = d;
            tmp_idx_v.first .push_back(V);
            tmp_idx_v.second.push_back(INF8);
            for (size_t i = 0; i < inEdges[v].size(); ++i) {
              int w = inEdges[v][i];
              if (!vis[w]) {
                que[que_h++] = w;
                vis[w] = true;
              }
            }
        pruned1:
            {}
          }

          que_t0 = que_t1;
          que_t1 = que_h;
        }

        for (int i = 0; i < que_h; ++i) vis[que[i]] = false;
        for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
          dst_r[tmp_idx_r.first[i]] = INF8;
        }
      }
      in_usd[r] = true;

      // out: forward
      {
        outLabel &idx_r = outIndex[inv[r]];
        const std::pair<std::vector<int>, std::vector<uint8_t> >
            &tmp_idx_r = tmp_out_idx[r];
        for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
          dst_r[tmp_idx_r.first[i]] = tmp_idx_r.second[i];
        }

        int que_t0 = 0, que_t1 = 0, que_h = 0;
        que[que_h++] = r;
        vis[r] = true;
        que_t1 = que_h;

        for (int d = 0; que_t0 < que_h; ++d) {
          for (int que_i = que_t0; que_i < que_t1; ++que_i) {
            int v = que[que_i];
            std::pair<std::vector<int>, std::vector<uint8_t> >
                &tmp_idx_v = tmp_in_idx[v];
            inLabel &idx_v = inIndex[inv[v]];

            // Prefetch
            _mm_prefetch(&idx_v.bpspt_d[0], _MM_HINT_T0);
            _mm_prefetch(&idx_v.bpspt_s[0][0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v.first[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v.second[0], _MM_HINT_T0);

            // Prune?
            if (out_usd[v]) continue;
            for (int i = 0; i < kNumBitParallelRoots; ++i) {
              int td = idx_r.bpspt_d[i] + idx_v.bpspt_d[i];
              if (td - 2 <= d) {
                td +=
                    (idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2 :
                    ((idx_r.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
                    (idx_r.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                    ? -1 : 0;
                if (td <= d) goto pruned2;
              }
            }


            visited_space += 1;
            for (size_t i = 0; i < tmp_idx_v.first.size(); ++i) {
              int w = tmp_idx_v.first[i];
              int td = tmp_idx_v.second[i] + dst_r[w];
              if (td <= d) goto pruned2;
            }

            // Traverse
            tmp_idx_v.first .back() = r;
            tmp_idx_v.second.back() = d;
            tmp_idx_v.first .push_back(V);
            tmp_idx_v.second.push_back(INF8);
            for (size_t i = 0; i < outEdges[v].size(); ++i) {
              int w = outEdges[v][i];
              if (!vis[w]) {
                que[que_h++] = w;
                vis[w] = true;
              }
            }
        pruned2:
            {}
          }

          que_t0 = que_t1;
          que_t1 = que_h;
        }

        for (int i = 0; i < que_h; ++i) vis[que[i]] = false;
        for (size_t i = 0; i < tmp_idx_r.first.size(); ++i) {
          dst_r[tmp_idx_r.first[i]] = INF8;
        }
      }
      out_usd[r] = true;
    }

    for (int v = 0; v < V; ++v) {
      int k = tmp_in_idx[v].first.size();
      // cout << k << endl;
      search_space += k;
      spt_v_memory += k * sizeof(uint32_t);
      spt_d_memory += k * sizeof(uint8_t);
      inIndex[inv[v]].num_items += k;
      inIndex[inv[v]].spt_v = (uint32_t*)memalign(64, k * sizeof(uint32_t));
      inIndex[inv[v]].spt_d = (uint8_t *)memalign(64, k * sizeof(uint8_t ));
      if (!inIndex[inv[v]].spt_v || !inIndex[inv[v]].spt_d) {
        Free();
        return false;
      }
      for (int i = 0; i < k; ++i) inIndex[inv[v]].spt_v[i] = tmp_in_idx[v].first[i];
      for (int i = 0; i < k; ++i) inIndex[inv[v]].spt_d[i] = tmp_in_idx[v].second[i];
      tmp_in_idx[v].first.clear();
      tmp_in_idx[v].second.clear();
    }

    for (int v = 0; v < V; ++v) {
      int k = tmp_out_idx[v].first.size();
      search_space +=k;
      spt_v_memory += k * sizeof(uint32_t);
      spt_d_memory += k * sizeof(uint8_t);
      // cout << k << endl;
      outIndex[inv[v]].num_items += k;
      outIndex[inv[v]].spt_v = (uint32_t*)memalign(64, k * sizeof(uint32_t));
      outIndex[inv[v]].spt_d = (uint8_t *)memalign(64, k * sizeof(uint8_t ));
      if (!outIndex[inv[v]].spt_v || !outIndex[inv[v]].spt_d) {
        Free();
        return false;
      }
      for (int i = 0; i < k; ++i) outIndex[inv[v]].spt_v[i] = tmp_out_idx[v].first[i];
      for (int i = 0; i < k; ++i) outIndex[inv[v]].spt_d[i] = tmp_out_idx[v].second[i];
      tmp_out_idx[v].first.clear();
      tmp_out_idx[v].second.clear();
    }
    
  }

  time_indexing_ += GetCurrentTimeSec();
  return true;
}

template<int kNumBitParallelRoots>
int PrunedLandmarkLabeling<kNumBitParallelRoots>
::QueryDistance(int v, int w) {
  if (v >= num_v_ || w >= num_v_) return v == w ? 0 : INT_MAX;

  const outLabel &idx_v = outIndex[v];
  const inLabel &idx_w = inIndex[w];
  int d = INF8;

  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_d[0], _MM_HINT_T0);

  for (int i = 0; i < kNumBitParallelRoots; ++i) {
    int td = idx_v.bpspt_d[i] + idx_w.bpspt_d[i];
    if (td - 2 <= d) {
      td +=
          (idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][0]) ? -2 :
          ((idx_v.bpspt_s[i][0] & idx_w.bpspt_s[i][1]) | (idx_v.bpspt_s[i][1] & idx_w.bpspt_s[i][0]))
          ? -1 : 0;

      if (td < d) d = td;
    }
  }
  // cout << "BP: " << d << endl;
  for (int i1 = 0, i2 = 0; ; ) {
    int v1 = idx_v.spt_v[i1], v2 = idx_w.spt_v[i2];
    if (v1 == v2) {
      if (v1 == num_v_) break;  // Sentinel
      int td = idx_v.spt_d[i1] + idx_w.spt_d[i2];
      if (td < d) d = td;
      ++i1;
      ++i2;
    } else {
      i1 += v1 < v2 ? 1 : 0;
      i2 += v1 > v2 ? 1 : 0;
    }
  }
  // cout << "pruned: " << d << endl;

  if (d >= INF8 - 2) d = INT_MAX;
  return d;
}


template<int kNumBitParallelRoots>
void PrunedLandmarkLabeling<kNumBitParallelRoots>
::Free() {
  for (int v = 0; v < num_v_; ++v) {
    free(inIndex[v].spt_v);
    free(inIndex[v].spt_d);
    free(outIndex[v].spt_v);
    free(outIndex[v].spt_d);
  }
  free(inIndex);
  free(outIndex);
  inIndex = NULL;
  outIndex = NULL;
  num_v_ = 0;
}


template<int kNumBitParallelRoots>
std::string PrunedLandmarkLabeling<kNumBitParallelRoots>
::Statistics() {
  std::string stats;
  stats += "index building time:" + std::to_string(time_indexing_) + "s,";
  // stats += " search space:" + std::to_string(search_space) + "\n";
  size_t memory_consumption_byte = spt_d_memory + spt_v_memory;
  double memory_consumption_MB = static_cast<double>(memory_consumption_byte) / (1024*1024); 
  stats += " index size:" + std::to_string(memory_consumption_MB) + "MB";
  return stats;
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::StoreIndex(const char* filename) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) return false;

  // Write number of vertices
  ofs.write(reinterpret_cast<const char*>(&num_v_), sizeof(num_v_));

  // Write inIndex
  for (int v = 0; v < num_v_; ++v) {
    // Write bit-parallel data
    ofs.write(reinterpret_cast<const char*>(inIndex[v].bpspt_d), sizeof(inIndex[v].bpspt_d));
    ofs.write(reinterpret_cast<const char*>(inIndex[v].bpspt_s), sizeof(inIndex[v].bpspt_s));

    // Write num_items
    ofs.write(reinterpret_cast<const char*>(&inIndex[v].num_items), sizeof(inIndex[v].num_items));

    // Write SPT data
    int spt_size = 0;
    while (inIndex[v].spt_v[spt_size] != num_v_) spt_size++;
    spt_size++; // Include sentinel

    ofs.write(reinterpret_cast<const char*>(&spt_size), sizeof(spt_size));
    ofs.write(reinterpret_cast<const char*>(inIndex[v].spt_v), spt_size * sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(inIndex[v].spt_d), spt_size * sizeof(uint8_t));
  }

  // Write outIndex
  for (int v = 0; v < num_v_; ++v) {
    // Write bit-parallel data
    ofs.write(reinterpret_cast<const char*>(outIndex[v].bpspt_d), sizeof(outIndex[v].bpspt_d));
    ofs.write(reinterpret_cast<const char*>(outIndex[v].bpspt_s), sizeof(outIndex[v].bpspt_s));

    // Write num_items
    ofs.write(reinterpret_cast<const char*>(&outIndex[v].num_items), sizeof(outIndex[v].num_items));

    // Write SPT data
    int spt_size = 0;
    while (outIndex[v].spt_v[spt_size] != num_v_) spt_size++;
    spt_size++; // Include sentinel

    ofs.write(reinterpret_cast<const char*>(&spt_size), sizeof(spt_size));
    ofs.write(reinterpret_cast<const char*>(outIndex[v].spt_v), spt_size * sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(outIndex[v].spt_d), spt_size * sizeof(uint8_t));
  }

  return ofs.good();
}

template<int kNumBitParallelRoots>
bool PrunedLandmarkLabeling<kNumBitParallelRoots>
::LoadIndex(const char* filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) return false;

  // Free existing index if any
  Free();

  // Read number of vertices
  ifs.read(reinterpret_cast<char*>(&num_v_), sizeof(num_v_));
  if (!ifs) return false;

  // Allocate memory for inIndex and outIndex
  inIndex = (inLabel*)memalign(64, num_v_ * sizeof(inLabel));
  outIndex = (outLabel*)memalign(64, num_v_ * sizeof(outLabel));
  if (!inIndex || !outIndex) {
    Free();
    return false;
  }

  // Initialize pointers to NULL
  for (int v = 0; v < num_v_; ++v) {
    inIndex[v].spt_v = NULL;
    inIndex[v].spt_d = NULL;
    outIndex[v].spt_v = NULL;
    outIndex[v].spt_d = NULL;
  }

  // Read inIndex
  for (int v = 0; v < num_v_; ++v) {
    // Read bit-parallel data
    ifs.read(reinterpret_cast<char*>(inIndex[v].bpspt_d), sizeof(inIndex[v].bpspt_d));
    ifs.read(reinterpret_cast<char*>(inIndex[v].bpspt_s), sizeof(inIndex[v].bpspt_s));
    if (!ifs) {
      Free();
      return false;
    }

    // Read num_items
    ifs.read(reinterpret_cast<char*>(&inIndex[v].num_items), sizeof(inIndex[v].num_items));
    if (!ifs) {
      Free();
      return false;
    }

    // Read SPT data
    int spt_size;
    ifs.read(reinterpret_cast<char*>(&spt_size), sizeof(spt_size));
    if (!ifs) {
      Free();
      return false;
    }

    inIndex[v].spt_v = (uint32_t*)memalign(64, spt_size * sizeof(uint32_t));
    inIndex[v].spt_d = (uint8_t*)memalign(64, spt_size * sizeof(uint8_t));
    if (!inIndex[v].spt_v || !inIndex[v].spt_d) {
      Free();
      return false;
    }

    ifs.read(reinterpret_cast<char*>(inIndex[v].spt_v), spt_size * sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(inIndex[v].spt_d), spt_size * sizeof(uint8_t));
    if (!ifs) {
      Free();
      return false;
    }
  }

  // Read outIndex
  for (int v = 0; v < num_v_; ++v) {
    // Read bit-parallel data
    ifs.read(reinterpret_cast<char*>(outIndex[v].bpspt_d), sizeof(outIndex[v].bpspt_d));
    ifs.read(reinterpret_cast<char*>(outIndex[v].bpspt_s), sizeof(outIndex[v].bpspt_s));
    if (!ifs) {
      Free();
      return false;
    }

    // Read num_items
    ifs.read(reinterpret_cast<char*>(&outIndex[v].num_items), sizeof(outIndex[v].num_items));
    if (!ifs) {
      Free();
      return false;
    }

    // Read SPT data
    int spt_size;
    ifs.read(reinterpret_cast<char*>(&spt_size), sizeof(spt_size));
    if (!ifs) {
      Free();
      return false;
    }

    outIndex[v].spt_v = (uint32_t*)memalign(64, spt_size * sizeof(uint32_t));
    outIndex[v].spt_d = (uint8_t*)memalign(64, spt_size * sizeof(uint8_t));
    if (!outIndex[v].spt_v || !outIndex[v].spt_d) {
      Free();
      return false;
    }

    ifs.read(reinterpret_cast<char*>(outIndex[v].spt_v), spt_size * sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(outIndex[v].spt_d), spt_size * sizeof(uint8_t));
    if (!ifs) {
      Free();
      return false;
    }
  }

  return true;
}

#endif  // PRUNED_LANDMARK_LABELING_H_
