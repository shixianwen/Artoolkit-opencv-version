#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//*******************************************
//ARtoolkit area
#ifdef _WIN32
#  include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __APPLE__
#  include <GL/gl.h>
#  include <GL/glut.h>
#else
#  include <OpenGL/gl.h>
#  include <GLUT/glut.h>
#endif
#include <AR/ar.h>
#include <AR/gsub.h>
#include <AR/video.h>
#include <AR/config.h>
#include <AR/param.h>	
#include "ARMarkerSquare.h"
//*******************************************
//fps measurement area
#include <sys/timeb.h>
//*******************************************
//fake parameter data in order to full the ARtoolkit to create a ARhandler for me.
//but in the AR3DHandle *ar3DHandle which is useful to estimate the arpose which is the rotation and translational matrix
#define             CPARA_NAME       "/home/shixianwen/Downloads/SDKs/bin/Data/camera_para.dat"
// Marker detection.
ARHandle           *arHandle;
ARPattHandle       *arPattHandle;
static int         gARPattDetectionMode;
// Transformation matrix retrieval.
AR3DHandle         *ar3DHandle;
static int           useContPoseEstimation = TRUE;

ARGViewportHandle  *vp;

int                 xsize, ysize;
int                 flipMode = 0;
int                 patt_id;
double              patt_width = 80.0;
int                 count1 = 0;
char                fps[256];
char                errValue[256];
int                 distF = 0;
int                 contF = 0;
ARParamLT          *gCparamLT = NULL;

// Markers.
static ARMarkerSquare *markersSquare = NULL;
static int markersSquareCount = 0;
//*******************************************************************************************
//fps function part
#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
    || defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 
int CLOCK()
{
    return clock();
}
#endif

#if defined(unix)        || defined(__unix)      || defined(__unix__) \
    || defined(linux)       || defined(__linux)     || defined(__linux__) \
    || defined(sun)         || defined(__sun) \
    || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__) \
    || defined(__FreeBSD__) || defined __DragonFly__ \
    || defined(sgi)         || defined(__sgi) \
    || defined(__MACOSX__)  || defined(__APPLE__) \
    || defined(__CYGWIN__) 
int CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}
#endif

double _avgdur=0;
int _fpsstart=0;
double _avgfps=0;
double _fps1sec=0;

double avgdur(double newdur)
{
    _avgdur=0.98*_avgdur+0.02*newdur;
    return _avgdur;
}

double avgfps()
{
    if(CLOCK()-_fpsstart>1000)      
    {
        _fpsstart=CLOCK();
        _avgfps=0.7*_avgfps+0.3*_fps1sec;
        _fps1sec=0;
    }

    _fps1sec++;
    return _avgfps;
}

//*******************************************************************************************
//part for camera parameter init()
static void   init(){
    ARParam         cparam;
    const char markerConfigDataFilename[] = "/home/shixianwen/Downloads/SDKs/bin/Data/markers1.dat";
    AR_PIXEL_FORMAT pixFormat;
    xsize = 640;
    ysize =  480;
    arParamChangeSize( &cparam, xsize, ysize, &cparam );
    ARLOG("*** Camera Parameter ***\n");
    arParamDisp( &cparam );
    ARLOGi("Camera Parameter Name '%s'.\n", CPARA_NAME);
    if( arParamLoad(CPARA_NAME, 1, &cparam) < 0 ) {
        ARLOGe("Camera parameter load error !!\n");
        exit(0);
    }
    if ((gCparamLT = arParamLTCreate(&cparam, AR_PARAM_LT_DEFAULT_OFFSET)) == NULL) {
        ARLOGe("Error: arParamLTCreate.\n");
        exit(-1);
    }

    if( (arHandle=arCreateHandle(gCparamLT)) == NULL ) {
        ARLOGe("Error: arCreateHandle.\n");
        exit(0);
    }



    if( (ar3DHandle=ar3DCreateHandle(&cparam)) == NULL ) {
        ARLOGe("Error: ar3DCreateHandle.\n");
        exit(0);
    }
    //here the input is RGB so pixFormat = 1 listed by AR_PIXEL_FORMAT;
    pixFormat = AR_PIXEL_FORMAT_BGR;
    if( arSetPixelFormat(arHandle, pixFormat) < 0 ) {
        ARLOGe("Error: arSetPixelFormat.\n");
        exit(0);
    }
    if( (arPattHandle=arPattCreateHandle()) == NULL ) {
        ARLOGe("Error: arPattCreateHandle.\n");
        exit(0);
    }
    newMarkers(markerConfigDataFilename, arPattHandle, &markersSquare, &markersSquareCount, &gARPattDetectionMode);
    ARLOGi("markersSquareCount Marker count = %d\n", markersSquareCount);
    arPattAttach( arHandle, arPattHandle );
    arSetPatternDetectionMode(arHandle, AR_MATRIX_CODE_DETECTION);
    arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_HAMMING63);
}

//*************************************************************************************************
//*******************************************
static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}


string cascadeName;
string nestedCascadeName;

int main( int argc, const char** argv )
{   
    
    VideoCapture capture;
    Mat frame, image,frameCopy;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    double scale;

    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|../../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    //test the video frame
    inputName = "compressedC0008.mp4";
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(c))
            cout << "Capture from camera #" <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }
    }
    else
    {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }
    init();
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

        for(;;)
        {   

	    clock_t start=CLOCK();
            capture >> frame;
            if( frame.empty() )
                break;

            Mat frame1 = frame.clone();
	    ARUint8* dataPtr = frame1.data;
	    //cout<<"rows = "<<frameCopy.rows<<" cols = "<<frameCopy.cols <<endl;
	    //*********************************************************************************************
	    //here is the part for do the marker detection in the ARtoolkit
	    /*for (int countx = 0; countx < xsize; countx++){
		for(int county = 0; county < ysize; county++){			
			int temp = ((county * ysize)+countx) *3;// Figure out index in array
			//dataPtr[temp] = frameCopy.at<cv::Vec3b>(countx,county)[0];
            		//dataPtr[temp + 1] = frameCopy.at<cv::Vec3b>(countx,county)[1];
            		//dataPtr[temp + 2] = frameCopy.at<cv::Vec3b>(countx,county)[2];
			if(countx == 200 && county == 200){
				cout << countx <<" "<<county<<endl;
				//cout << dataPtr[temp] << " " << dataPtr[temp+1] << " " << dataPtr[temp+2]<<endl;
 				    ARLOG("%d dataPtr\n", dataPtr[temp]);
   			            ARLOG("%d dataPtr\n", dataPtr[temp + 1] );
    				    ARLOG("%d dataPtr\n", dataPtr[temp + 2]);	  
				    ARLOG("%d frameCopy\n", frameCopy.at<cv::Vec3b>(200,200)[0]);
   			            ARLOG("%d frameCopy\n", frameCopy.at<cv::Vec3b>(200,200)[1]);
    				    ARLOG("%d frameCopy\n", frameCopy.at<cv::Vec3b>(200,200)[2]);		
			}
		}	    
	    }*/
	    
           	ARMarkerInfo   *markerInfo;
  	        int             markerNum;
	        ARdouble        err;
	        if( arDetectMarker(arHandle, dataPtr) < 0 ) {
        		cout<<"cannot find Marker"<<endl;
    		}else{		
				//get Detected Marker
				//cout<<"find Marker#!@#$"<<endl;
				markerNum = arGetMarkerNum( arHandle );
				//ARLOGi("markerNum = %d\n",markerNum);
				//cout<<"find Marker"<<endl;
				markerInfo =  arGetMarker( arHandle );
				for(int i = 0; i < markersSquareCount;i++ ){
					markersSquare[i].validPrev = markersSquare[i].valid;
					int k = -1;
					if (markersSquare[i].patt_type == AR_PATTERN_TYPE_MATRIX) {
						for (int j = 0; j < markerNum; j++) {
							//ARLOGi("markersSquare[%d].patt_id = %d\n",i,markersSquare[i].patt_id);
							//ARLOGi("markerInfo[%d].id = %d\n",j,markerInfo[j].id);
							if (markersSquare[i].patt_id == markerInfo[j].id) {
								if (k == -1) {
									if (markerInfo[j].cfPatt >= markersSquare[i].matchingThreshold) k = j; // First marker detected.
								} else if (markerInfo[j].cfPatt > markerInfo[k].cfPatt) k = j; // Higher confidence marker detected.
							}
						}

					} 
					if( k != -1){
						markersSquare[i].valid = TRUE;
						//ARLOGi("Marker %d matched pattern %d.\n", i, markerInfo[k].id);					//get the transformation matrix from the markers to camera in camera coordinate
						//rotation matrix + translation matrix
						if (markersSquare[i].validPrev && useContPoseEstimation) {
							err = arGetTransMatSquareCont(ar3DHandle, &(markerInfo[k]), markersSquare[i].trans, markersSquare[i].marker_width, markersSquare[i].trans);
						} else {
							err = arGetTransMatSquare(ar3DHandle, &(markerInfo[k]), markersSquare[i].marker_width, markersSquare[i].trans);
						}
					     //error
					     cout<<"error=" << err<<endl;
					     //direction of the marker
					     cout<<"direction"<<markerInfo[k].dir<<endl;
					     //center of the marker on the screen
					     cout<<"\nmarkerInfo[k].pos[0] = " <<markerInfo[k].pos[0] << "\nmarkerInfo[k].pos[1] = "<<markerInfo[k].pos[1] <<"\n";	     
					     circle( frame1, Point( markerInfo[k].pos[0], markerInfo[k].pos[1] ), 5.0, Scalar( 0, 0, 255 ), 1, 8 );
					     for(int i1 = 0; i1 <4;i1++){
					     	Point *pt1 = new Point(markerInfo[k].vertex[i1][0], markerInfo[k].vertex[i1][1]);
                                                Point *pt2;
						if(i1<=2){
						  pt2 = new Point(markerInfo[k].vertex[i1+1][0], markerInfo[k].vertex[i1+1][1]);
						}else{
						  pt2 = new Point(markerInfo[k].vertex[0][0], markerInfo[k].vertex[0][1]);
						}
					        
					        line(frame1, *pt1, *pt2, Scalar(0,255,0), 3);
                                             }
					     
					     for(int i1=0;i1<3;i1++){
					     	
						cout<<markersSquare[i].trans[i1][3]<<endl;

					     }
					     //draw(markersSquare[i].trans);
					}else{
						markersSquare[i].valid = FALSE;					
					}
				}
			}
            int c = waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
	    double dur = CLOCK()-start;
            printf("avg time per frame %f ms. fps %f. \n",avgdur(dur),avgfps() );
	    
            imshow( "result", frame1 );

        }
    }
    return 0;
}

