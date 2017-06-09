FiraColorTracking
=================
### Proof of concept for a color blob follower lab exercise

Project implemented as a part of Programming of Autonomous Mobile Robots course at AGH University of Science and Technology.

### Warning: This repository might and will contain work in progress as well as my vain attempts and utter failures. Proceed at your own risk.

## Requirements

Android 4.1+ phone with camera, bluetooth, full battery and high-performance hardware 

## Setup 
1. Install OpenCV Manager on the device using Play Store or adb:

https://play.google.com/store/apps/details?id=org.opencv.engine&hl=en
or
http://sourceforge.net/projects/opencvlibrary/files/opencv-android/3.0.0/OpenCV-3.0.0-android-sdk-1.zip/download

Get device architecture:

`adb shell getprop ro.product.cpu.abi`

Install  architecture:

`<AndroidSDK>/platform-tools/adb install <OpenCVSDK>/apk/OpenCV_3.0.0_Manager(...).apk`

when in doubt, see: http://docs.opencv.org/3.0.0/da/d2a/tutorial_O4A_SDK.html

2. Open the project in Android Studio.
3. Install all required SDK and tools as prompted by the IDE.
**Do not upgrade Gradle version in the project** - this would imply a higher SDK version requirement.
4. (Unix/Windows) CTRL+Shift+A -> SDK Manager -> SDK Tools -> choose "ConstraintLayout for Android" and "Solver for ConstraintLayout" 
5. Plug your device and run (Shift+F10).
6. Pairing with FIRA can be troublesome. 
What works (usually) is to pair manually in Settings and then again in the app. 
Watch your notifications bar - the robot won't usually pair directly, 
but it will issue its own pairing request to your device.
After successful pairing you will see a popup with robot bluetooth ID.
7. Implement robot movement in pl.robolab.fira.colortracking.MainActivity#onCenterPoint or just run the example.

## Usage

* **VOLUME_UP** - movement test enable/disable
* **BACK** - send stop command, exit on second
* **MENU** - bluetooth command. On newer devices
* touch - pick color range for tracking

## Description

Provided application follows a very simple routine.
For each frame: 
* 2x Imgproc.pyrDown - blurs an image and downsamples it. The function performs the downsampling step of the Gaussian pyramid construction.
* Imgproc.cvtColor - converts to HSV.
* Core.inRange - creates mask from lower to upper color bound.
* Imgproc.dilate.
* Imgproc.findContours with flags:
CV_RETR_EXTERNAL retrieves only the extreme outer contours
CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an upper-right rectangular contour is encoded with 4 points.
* Sort contours by size.
* Filter to 0.1 * maxArea.
* Imgproc.convexHull - The functions find the convex hull of a 2D point set using the Sklansky’s algorithm [Sklansky82] that has O(N logN) complexity in the current implementation.
* Circle the middle point of the hull.
* Execute user provided callback (pl.robolab.fira.colortracking.MainActivity#onCenterPoint)
