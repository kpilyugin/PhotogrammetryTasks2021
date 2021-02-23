#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"

phg::FlannMatcher::FlannMatcher()
{
    const int num_trees = 4;
    const int num_checks = 32;

    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(num_trees);
    search_params = flannKsTreeSearchParams(num_checks);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices;
    cv::Mat distances;
    flann_index->knnSearch(query_desc, indices, distances, 2, *search_params);
    int numMatches = indices.rows;
    for (size_t i = 0; i < numMatches; ++i) {
        int i1 = indices.at<int>(i, 0);
        int i2 = indices.at<int>(i, 1);
        float d1 = distances.at<float>(i, 0);
        float d2 = distances.at<float>(i, 1);
        matches.push_back({
            cv::DMatch(i, i1, d1), cv::DMatch(i, i2, d2)
        });
    }
}
