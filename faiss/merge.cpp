/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/merge.h>

#include <faiss/IndexPreTransform.h>
#include <faiss/MetaIndexes.h>

namespace faiss {
namespace merge{
void check_compatible_for_merge(const Index* index0, const Index* index1){
    const faiss::IndexPreTransform* pt0 =
            dynamic_cast<const faiss::IndexPreTransform*>(index0);

    if (pt0) {
        const faiss::IndexPreTransform* pt1 =
                dynamic_cast<const faiss::IndexPreTransform*>(index1);
        FAISS_THROW_IF_NOT_MSG(pt1, "both indexes should be pretransforms");

        FAISS_THROW_IF_NOT(pt0->chain.size() == pt1->chain.size());
        for (int i = 0; i < pt0->chain.size(); i++) {
            FAISS_THROW_IF_NOT(typeid(pt0->chain[i]) == typeid(pt1->chain[i]));
        }

        index0 = pt0->index;
        index1 = pt1->index;
    }
    FAISS_THROW_IF_NOT(typeid(index0) == typeid(index1));
    FAISS_THROW_IF_NOT(
            index0->d == index1->d &&
            index0->metric_type == index1->metric_type);

    
    index0->check_compatible_for_merge(*index1);
    /*const faiss::IndexIVF* ivf0 = dynamic_cast<const faiss::IndexIVF*>(index0);
    if (ivf0) {
        const faiss::IndexIVF* ivf1 =
                dynamic_cast<const faiss::IndexIVF*>(index1);
        FAISS_THROW_IF_NOT(ivf1);

        ivf0->check_compatible_for_merge(*ivf1);
    }*/

    // TODO: check as thoroughfully for other index types
}

const Index* try_extract_index(const Index* index) {
    if (auto* pt = dynamic_cast<const IndexPreTransform*>(index)) {
        index = pt->index;
    }

    if (auto* idmap = dynamic_cast<const IndexIDMap*>(index)) {
        index = idmap->index;
    }
    if (auto* idmap = dynamic_cast<const IndexIDMap2*>(index)) {
        index = idmap->index;
    }

    //auto* ivf = dynamic_cast<const IndexIVF*>(index);

    return index;
}

Index* try_extract_index(Index* index) {
    return const_cast<Index*>(try_extract_index((const Index*)(index)));
}


const Index* extract_index(const Index* index) {
    const Index* ivf = try_extract_index(index);
    FAISS_THROW_IF_NOT(ivf);
    return ivf;
}

Index* extract_index(Index* index) {
    return const_cast<Index*>(extract_index((const Index*)(index)));
}

void merge_into(faiss::Index* index0, faiss::Index* index1, bool shift_ids) {
    check_compatible_for_merge(index0, index1);
    Index* ivf0 = extract_index(index0);
    Index* ivf1 = extract_index(index1);

    ivf0->merge_from(*ivf1, shift_ids ? ivf0->ntotal : 0);

    // useful for IndexPreTransform
    index0->ntotal = ivf0->ntotal;
    index1->ntotal = ivf1->ntotal;
}
}
}
