
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include <unordered_set>

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1024, 640);

    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> tempKptMatches;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1 ++)
    {
        cv::Point2f tempPtPre;
        cv::Point2f tempPtCur;

        tempPtCur = kptsCurr.at(it1->trainIdx).pt;
        tempPtPre = kptsPrev.at(it1->queryIdx).pt;

        if (boundingBox.roi.contains(tempPtCur) && boundingBox.roi.contains(tempPtPre))
        {
            tempKptMatches.push_back(*it1);
        }
    }

    boundingBox.kptMatches = tempKptMatches;
}

static void filterKptMatches(std::vector<cv::DMatch> kptMatches, std::vector<cv::DMatch> & qualifiedkptMatches)
{
    double sum = 0.0;
    double mean = 0.0;
    double temp = 0.0;
    double sd = 0.0;

    // process pre lidar points first
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    {
        sum += it1->distance;
    }
    mean = sum / kptMatches.size();

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    {
        temp += pow(it1->distance - mean, 2);
    }

    sd = sqrt(temp / kptMatches.size());

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    {
        temp = std::abs(it1->distance - mean);
        if (temp < (sd))
        {
            qualifiedkptMatches.push_back(*it1);
        }
    }

    return;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<cv::DMatch> qualifiedkptMatches;
    double dT = 0.0;

    // process 
    filterKptMatches(kptMatches, qualifiedkptMatches);
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = qualifiedkptMatches.begin(); it1 != qualifiedkptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = qualifiedkptMatches.begin() + 1; it2 != qualifiedkptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

    std::cout << "=============TEST camera TTC: " << TTC << std::endl;
}

static void filterPoints(std::vector<LidarPoint> & lidarPoints, std::vector<LidarPoint> & qualifiedPoint)
{
    double sum = 0.0;
    double mean = 0.0;
    double temp = 0.0;
    double sd = 0.0;

    // process pre lidar points first
    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); it1++)
    {
        sum += it1->x;
    }
    mean = sum / lidarPoints.size();

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); it1++)
    {
        temp += pow(it1->x - mean, 2);
    }

    sd = sqrt(temp / lidarPoints.size());

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); it1++)
    {
        temp = std::abs(it1->x - mean);
        if (temp < (3 * sd))
        {
            qualifiedPoint.push_back(*it1);
        }
    }

    return;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    std::vector<LidarPoint> qualifiedPointPre;
    std::vector<LidarPoint> qualifiedPointCur;
    double minCur = 1000.0;
    double minPre = 1000.0;

    // process cur lidar points then
    filterPoints(lidarPointsCurr, qualifiedPointCur);
    for (auto it1 = qualifiedPointCur.begin(); it1 != qualifiedPointCur.end(); it1 ++)
    {
        if (it1->x < minCur)
        {
            minCur = it1->x;
        }
    }
    std::cout << "=============TEST lidar minCur: " << minCur << std::endl;

    // then process previous one
    filterPoints(lidarPointsPrev, qualifiedPointPre);
    for (auto it1 = qualifiedPointPre.begin(); it1 != qualifiedPointPre.end(); it1 ++)
    {
        if (it1->x < minPre)
        {
            minPre = it1->x;
        }
    }
    std::cout << "=============TEST lidar minPre: " << minPre << std::endl;

    // calculate the ttc
    TTC = minCur / ((minPre - minCur) * frameRate);

    return;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::map<int, int> prePoint2Box; // (point, box)
    std::map<int, int> curPoint2Box; // (point, box)
    std::vector<std::tuple<int, int>> tempTuple;

    cv::Point2f tempPt;
    std::unordered_set<int> preProcessSet;
    std::unordered_set<int> curProcessSet;

    // special treatement for 1st frame, need to handle previous frame.
    // otherwise only need to handle current frame, as previous already done in previous loop
    if (prevFrame.boundingBoxes.at(0).keypoints.size() == 0)
    {
        for (auto it1 = prevFrame.keypoints.begin(); it1 != prevFrame.keypoints.end(); it1++)
        {
            tempPt = it1->pt;
            // find matches for this point on all possible box
            for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); it2++)
            {
                // find a keypoint to bounding box matches
                if (it2->roi.contains(tempPt))
                {
                    // preserve result to buffer
                    it2->keypoints.push_back(*it1);
                    break;
                }
            }
        }
    }

    // normal process for current frame, fill keypoints in each box
    for (auto it1 = currFrame.keypoints.begin(); it1 != currFrame.keypoints.end(); it1++)
    {
        tempPt = it1->pt;
        // find matches for this point on all possible box
        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); it2++)
        {
            // find a keypoint to bounding box matches
            if (it2->roi.contains(tempPt))
            {
                // preserve result to temp buffer
                it2->keypoints.push_back(*it1);
                break;
            }
        }
    }

    // loop through matches then
    for (auto it1 = matches.begin(); it1 != matches.end(); it1++)
    {
        auto prekptIdx = it1->queryIdx;
        auto curkptIdx = it1->trainIdx;

        int preMatch, curMatch;
        bool preMatchFlag = false;
        bool curMatchFlag = false;

        // now do normal matching
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); it2++)
        {
            tempPt = prevFrame.keypoints.at(prekptIdx).pt;
            // find the keypoint to bounding box matches
            if (it2->roi.contains(tempPt))
            {
                preMatch = it2->boxID;
                preMatchFlag = true;
                break;
            }
        }

        // only process when matches within previous boxes
        if (preMatchFlag == true)
        {
            // now do normal matching
            for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); it2++)
            {
                tempPt = currFrame.keypoints.at(curkptIdx).pt;
                // find the keypoint to bounding box matches
                if (it2->roi.contains(tempPt))
                {
                    curMatch = it2->boxID;
                    curMatchFlag = true;
                    break;
                }
            }

            // only process when 2 match exist
            if (curMatchFlag == true)
            {
                tempTuple.push_back(std::tuple<int, int>(preMatch, curMatch));
            }
        }
    }

    // final process, generate the map
    for (auto it1 = tempTuple.begin(); it1 != tempTuple.end(); it1++)
    {
        auto preBoxId = std::get<0>(*it1);

        if (preProcessSet.find(preBoxId) == preProcessSet.end())
        {
            std::vector<int> tempCounter;

            // not processed before, do the search
            preProcessSet.insert(preBoxId);

            // loop again to find pairs
            for (auto it2 = tempTuple.begin(); it2 != tempTuple.end(); it2++)
            {
                if (std::get<0>(*it1) == preBoxId)
                {
                    tempCounter.push_back(std::get<1>(*it1));
                }
            }

            // now find the maximum likelihood
            int max = 0;
            int mostCommon = -1;
            map<int, int> maxSortMap;
            for (auto it2 = tempCounter.begin(); it2 != tempCounter.end(); it2++)
            {
                maxSortMap[*it2]++;
                if (maxSortMap[*it2] > max)
                {
                    max = maxSortMap[*it2];
                    mostCommon = *it2;
                }
            }

            bbBestMatches[preBoxId] = mostCommon;
        }
    }
}
