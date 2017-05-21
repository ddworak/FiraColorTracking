package pl.robolab.fira.colortracking;

import android.util.Log;
import android.util.Pair;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ColorBlobDetector {
    // Minimum contour area in percent for contours filtering
    private static final double sMinContourArea = 0.1;
    // Color radius for range checking in HSV color space
    private static final Scalar sColorRadius = new Scalar(25, 50, 50, 0);
    private static final Comparator<Pair<Double, MatOfPoint>> sByContourSizeComparator = (lhs, rhs) -> Double.compare(lhs.first, rhs.first);
    // Lower and Upper bounds for range checking in HSV color space
    private final Scalar mLowerBound = new Scalar(0);
    private final Scalar mUpperBound = new Scalar(0);
    private final Mat mSpectrum = new Mat();
    private final List<MatOfPoint> mHull = new ArrayList<>();

    // Cache
    private final Mat mPyrDownMat = new Mat();
    private final Mat mHsvMat = new Mat();
    private final Mat mMask = new Mat();
    private final Mat mDilatedMask = new Mat();
    private final Mat mHierarchy = new Mat();


    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.val[0] >= sColorRadius.val[0]) ? hsvColor.val[0] - sColorRadius.val[0] : 0;
        double maxH = (hsvColor.val[0] + sColorRadius.val[0] <= 255) ? hsvColor.val[0] + sColorRadius.val[0] : 255;

        mLowerBound.val[0] = minH;
        mUpperBound.val[0] = maxH;

        mLowerBound.val[1] = hsvColor.val[1] - sColorRadius.val[1];
        mUpperBound.val[1] = hsvColor.val[1] + sColorRadius.val[1];

        mLowerBound.val[2] = hsvColor.val[2] - sColorRadius.val[2];
        mUpperBound.val[2] = hsvColor.val[2] + sColorRadius.val[2];

        mLowerBound.val[3] = 0;
        mUpperBound.val[3] = 255;

        Mat spectrumHsv = new Mat(1, (int) (maxH - minH), CvType.CV_8UC3);

        for (int j = 0; j < maxH - minH; j++) {
            byte[] tmp = {(byte) (minH + j), (byte) 255, (byte) 255};
            spectrumHsv.put(0, j, tmp);
        }

        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public void process(Mat rgbaImage) {
        Imgproc.pyrDown(rgbaImage, mPyrDownMat);
        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);

        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);

        Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
        Imgproc.dilate(mMask, mDilatedMask, new Mat());

        ArrayList<MatOfPoint> allContours = new ArrayList<>();

        Imgproc.findContours(mDilatedMask, allContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        ArrayList<Pair<Double, MatOfPoint>> sizedContours = new ArrayList<>();
        for (MatOfPoint contour : allContours) {
            sizedContours.add(Pair.create(Imgproc.contourArea(contour), contour));
        }
        Collections.sort(sizedContours, sByContourSizeComparator);
        mHull.clear();
        if (!sizedContours.isEmpty()) {
            final double maxArea = sizedContours.get(0).first;
            MatOfPoint hullPointMat = new MatOfPoint();
            MatOfInt hull = new MatOfInt();

            // Filter contours by area and resize to fit the original image size, calculating convex hull
            for (int i = 0; i < sizedContours.size() && sizedContours.get(i).first > sMinContourArea * maxArea; i++) {
                MatOfPoint contour = sizedContours.get(i).second;
                Core.multiply(contour, new Scalar(4, 4), contour);
                Imgproc.convexHull(contour, hull);
                List<Point> hullPointList = new ArrayList<>();
                List<Point> contourPoints = contour.toList();
                List<Integer> hullIndices = hull.toList();
                for (int j = 0; j < hullIndices.size(); j++) {
                    hullPointList.add(contourPoints.get(hullIndices.get(j)));
                }
                hullPointMat.fromList(hullPointList);
            }
            Log.e("lol", "" + hullPointMat.size().toString());
            mHull.clear();
            mHull.add(hullPointMat);
        }
    }

    public List<MatOfPoint> getHull() {
        return mHull;
    }
}
