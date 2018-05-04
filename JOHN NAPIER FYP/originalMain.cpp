//  #### filenames updated to cater for neural network training ####
//  main.cpp
//  Final_Program
//
//  Created by john napier on 11/04/2017.
//  Copyright © 2017 john napier. All rights reserved.
//
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;
//Global Variables (Filtering)
Mat src; Mat dst; Mat dst2;

//flag to see if contor was drawn
int drawnContour = -1;
//Global Variables (Skull Stripping)
int threshold_value = 0;
int threshold_type = 3;;
int const max_BINARY_value = 255;

Mat src_ss, dst_ss, src_gray, src2, src_gray2;
vector<vector<Point> > contours;
int erosion_elem = 0;
int erosion_size = 0;

int erosion_elem2 = 0;
int erosion_size2 = 1;
int dilation_elem = 0;
int dilation_size = 11;
int dilation_elem2 = 0;
int dilation_size2 = 4;

//Global Variables (Haemorrhage Detection)
Mat showThresh, gray, grayerode;
int extaAxial = 0;
float intraCranial = 0;
double hull = 0;
Mat copyOrig;

int thresh = 50;
int false_Positive_Count = 0;
int max_Haemorrhage_Count = 0;
int Haemorrhage_Count = 0;
int max_thresh = 255;
int image_counter = 1;
int erosion_elem_Haem = 0;
int erosion_size_Haem = 1;



//Methods All
void ThresholdFinal( int, void* );
void ThresholdKeepSkull( int, void* );
Mat segment(Mat original, int flag);
void Dilation( int, void* );
void Erosion( int, void* );
void Dilation2( int, void* );
void Erosion2( int, void* );
int findLargestDensity(vector< vector<Point> > contours);
Mat thresh_callback(int, void* );
void Erosion_Haem( int, void* );






//Functions
void Filtering();
void Skull_Stripping();
void Haemorrhage_Detection();

int main(int argc, const char * argv[]) {
    Filtering();
    Skull_Stripping();
    Haemorrhage_Detection();
    return 0;
}

void Filtering()
{
    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    String folder = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/Testing/new_bleed4Contrast"; // again we are using the Opencv's embedded "String" class
    
    glob(folder, filenames);
    stringstream ss; //to hold image name;
    
    string name = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/Testing/testBilateral/bilateral_";
    string type = ".png";
    int ct = 0;
    
    
    for(size_t j = 0; j < filenames.size(); ++j)
    {
        src = imread(filenames[j]);
        
        if(!src.data)
            cerr << "Problem loading image!!!" << filenames[j] << endl ;
        else
        {
            
            
            dst = src.clone();
            //if( display_dst(  ) != 0 ) { return 0; }
            //display_dst("Original Image");
            
            
            /// Applying Median blur
            //for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
            int i = 5;
            bilateralFilter ( src, dst, i, i*2, i/2 );
            //if( display_dst(  ) != 0 ) { return 0; }}
            // display_dst("filtered Image");
            
            if (ct < 9)
                ss<<name<<"0"<<(ct += 1)<<type; //joining name and number.
            else
                ss<<name<<(ct += 1)<<type; //joining name and number.
            //ss<<name<<(ct += 1)<<type; //joining name and number.
            
            string filename = ss.str();
            ss.str("");
            imwrite(filename, dst); // saving image
        }
        
    }

}

void Skull_Stripping()
{
    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    String folder = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/Testing/testBilateral"; // again we are using the Opencv's embedded "String" class
    
    vector<String> filenames2; // notice here that we are using the Opencv's embedded "String" class
    String folder2 = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/SegmentedTissue"; // again we are using the Opencv's embedded "String" class
    
    glob(folder, filenames);
    glob(folder2, filenames2);
    
    stringstream ss; //to hold image name;
    stringstream ss2; //to hold image name;
    stringstream ss3; //to hold image name;
    stringstream ss4; //to hold image name;
    
    
    
    
    string name = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/InitalTrheshold/threshold_";//location to store the removed skull
    string name2 = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/SegmentedSkull/SegmentedSkull_";//loacation to store the final image
    string name3 = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/RemovedTissue/RemovedTissue_";//location to store the removed tissue
    string name4 = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/SegmentedTissue/SegmentedTissue_"; //location to store the segmented head
    string type = ".jpg";
    int ct = 0; //image counters
    int ct2 = 0;
    int ct3 = 0;
    int ct4 = 0;
    
    
    for(size_t j = 0; j < filenames.size(); ++j)
    {
        
        int flag = 0; // contour detection technique
        
        // Load an image
        src_ss = imread(filenames[j],1);
        
        if(! src_ss.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << filenames[j] << endl ;
        }
        
        else //If inputed file is valid
        {
            //src_ss = cvLoadImage(writable);//imreald(filenames[j],1);
            
            //imshow("show", src_ss);
            //waitKey();
            src_gray = src_ss;
            /// Convert the image to Gray
            cvtColor( src_ss, src_gray, CV_BGR2GRAY );
            
            
            //reading the segmented skull image to use as a reference
            src2 = imread(filenames2[j],1);
            
            
            /// Create a window to display results
            namedWindow("Initial Threshold", CV_WINDOW_AUTOSIZE );
            
            //Removing head tissue and keeping the skull
            ThresholdKeepSkull( 0, 0 );
            
            if (ct3 < 9)
                ss3<<name3<<"0"<<(ct3 += 1)<<type; //joining name and number.
            else
                ss3<<name3<<(ct3 += 1)<<type; //joining name and number.
            //ss3<<name3<<(ct3 += 1)<<type; //joining name and number.
            
            string filename3 = ss3.str();
            ss3.str("");
            imwrite(filename3, dst_ss); // saving image
            
            
            //Segmented Head with skull
            Mat tissue = segment(src_gray, flag);
            if (ct4 < 9)
                ss4<<name4<<"0"<<(ct4 += 1)<<type; //joining name and number.
            else
                ss4<<name4<<(ct4 += 1)<<type; //joining name and number.
            //ss4<<name4<<(ct4 += 1)<<type; //joining name and number.
            
            string filename4 = ss4.str();
            ss4.str("");
            imwrite(filename4, tissue); // saving image
            //cvtColor( tissue, dst_ss, CV_GRAY2BGR );
            dst_ss = tissue;
            src_gray2 = dst_ss;
            
            
            /// Removing Skull (removing white)
            ThresholdFinal( 0, 0 );
            
            if (ct < 9)
                ss<<name<<"0"<<(ct += 1)<<type; //joining name and number.
            else
                ss<<name<<(ct += 1)<<type; //joining name and number.
            //ss<<name<<(ct += 1)<<type; //joining name and number.
            
            string filename = ss.str();
            ss.str("");
            imwrite(filename, dst_ss); // saving image
            
            flag = 0;
            
            //Remove skull and obtain the final brain mask
            Mat skull = segment(src_gray, flag);
            if (ct2 < 9)
                ss2<<name2<<"0"<<(ct2 += 1)<<type; //joining name and number.
            else
                ss2<<name2<<(ct2 += 1)<<type; //joining name and number.
            
            string filename2 = ss2.str();
            ss2.str("");
            imwrite(filename2, skull); // saving image
            
            //delete[] writable;
        }
        
        
        
    }

}

void ThresholdFinal( int, void* ) //removing skull
{
    //Erosion2(0, 0);
    threshold( src_gray2, dst_ss, 245, max_BINARY_value,4 );
    
    Erosion2(0, 0);
    Dilation2( 0, 0 );
    //threshold( dst, dst, 10, 4,3 );
    
    
    imshow( "Initial Threshold", dst_ss );
    // waitKey();
}
void ThresholdKeepSkull( int, void* ) //removing head tissue and brain WM and GM
{
    
    threshold( src_gray, dst_ss, 254, max_BINARY_value,3 );
    Dilation( 0, 0 );
    Erosion(0, 0);
    imshow( "White Threshold", dst_ss );
    // waitKey();
}


Mat segment(Mat orignalImage, int flag) //Segmentation
{
    // read in the apple (change path to the file)
    Mat img0 = dst_ss;
    Mat original = orignalImage;
    
    Mat img1;
    //cvtColor(img0, img1, CV_RGB2GRAY);
    img1 = img0;
    
    
    // find the contours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat threshold_output;
    int thresh = 50;
    threshold( img1, threshold_output, thresh, 255, THRESH_BINARY );
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    // you could also reuse img1 here
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
    
    
    
    /*
     Before drawing all contours you could also decide
     to only draw the contour of the largest connected component
     found. Here's some commented out code how to do that:
     */
    
    vector<double> areas(contours.size());
    
    //to calculate contour points
    vector<double> maxPoints(contours.size());
    vector<double> minPoints(contours.size());
    
    /* for(int i = 0; i < contours.size(); i++)
     {
     maxPoints[i] = contours[i];
     }*/
    vector<vector<Point> > contours_poly( contours.size() );
    Scalar color;
    RNG rng(12345);
    if (flag == 0)
    {
        
        for(int i = 0; i < contours.size(); i++)
        {
            //cout<<contours[i]<<"-----"<<endl;
            areas[i] = contourArea(Mat(contours[i]));
        }
        double max;
        Point maxPosition;
        minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
        drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);
    }
    else if (flag == 1)
    {
        
        int largest = findLargestDensity(contours);
        drawContours(mask, contours, largest, Scalar(1),CV_FILLED);
        
    }
    
    // let's create a new image now
    Mat crop(original.rows, original.cols, CV_8UC3);
    
    // set background to black
    crop.setTo(Scalar(0,0,0));
    
    // and copy the brain
    
    original.copyTo(crop, mask);
    
    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
    
    // show the images
    imshow("original", img0);
    imshow("mask", mask);
    imshow("canny", img1);
    imshow("cropped", crop);
    
    return crop;
}

void Dilation( int, void* ) //Dilation for skull
{
    int dilation_type;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( dilation_type,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    dilate( dst_ss, dst_ss, element );
    imshow( "Dilation Demo", dst_ss );
}

void Dilation2( int, void* )//Dilation for final image
{
    int dilation_type;
    if( dilation_elem2 == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem2 == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem2 == 2) { dilation_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( dilation_type,
                                        Size( 2*dilation_size2 + 1, 2*dilation_size2+1 ),
                                        Point( dilation_size2, dilation_size2) );
    /// Apply the dilation operation
    dilate( dst_ss, dst_ss, element );
    imshow( "Dilation Demo", dst_ss );
}

void Erosion( int, void* ) //Erosion for skull
{
    int erosion_type;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    /// Apply the erosion operation
    erode( dst_ss, dst_ss, element );
    imshow( "Erosion Demo", dst_ss );
}

void Erosion2( int, void* )//Erosion for final image
{
    int erosion_type;
    if( erosion_elem2 == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem2 == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem2 == 2) { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size2 + 1, 2*erosion_size2+1 ),
                                        Point( erosion_size2, erosion_size2 ) );
    
    /// Apply the erosion operation
    erode( dst_ss, dst_ss, element );
    imshow( "Erosion Demo", dst_ss );
}



int findLargestDensity(vector< vector<Point> > contours) //Used to find contour with most amount of pixels.
{
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    double maxArea = 0;
    int maxCountour = 0;;
    
    for( int i = 0; i < contours.size(); i++ )
    { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        if ((boundRect[i].width * boundRect[i].height )> maxArea)
        {
            maxArea =boundRect[i].width * boundRect[i].height;
            maxCountour = i;
        }
    }
    return maxCountour;
}

void Haemorrhage_Detection()
{
    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    String folder = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/SegmentedSkull"; // again we are using the Opencv's embedded "String" class
    glob(folder, filenames);
    stringstream ss; //to hold image name;
    string name = "/home/kirsty/Desktop/fyp/fyp/JOHN NAPIER FYP/Haemorrhage/haemorrhage_";//location to store the removed skull
    string type = ".png";
    int ct = 0; //image counters
    
    //saving grayscale and contour image
    stringstream ss2; //to hold image name;
    string name2 = "/home/kirsty/Desktop/fyp/fyp/classification/Testing/grayscale/gs_";//location to store the removed skull
    string type2 = ".png";

    stringstream ss3; //to hold image name;
    string name3 = "/home/kirsty/Desktop/fyp/fyp/classification/Testing/contour/c_";//location to store the removed skull
    string type3 = ".png";
    
    for(size_t j = 0; j < filenames.size(); ++j)//looping thriugh all the files
    {
        gray=imread(filenames[j],0);// original
        copyOrig = imread(filenames[j],0);// used for further processing
        grayerode  =  imread(filenames[j],0);//used in eroding stage
        
        
        
        if(! gray.data ) // Check for invalid input
        {
            cout <<  "Could not open or find the image" << filenames[j] << endl ;
        }
        
        else // if the input is valid
        {
            
            
            namedWindow( "Gray", 1 );
            
            // Initialize parameters
            int histSize = 256;    // bin size
            int histSize2 = 4; //dividing the intensities into 4 clusters
            float range[] = { 1, 255 }; //the range of grayscale intensities which need to be considered
            const float *ranges[] = { range };
            
            // Calculate histogram
            MatND hist;
            MatND hist2;
            
            calcHist( &gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
            calcHist( &gray, 1, 0, Mat(), hist2, 1, &histSize2, ranges, true, false );
            
            double counter = 0;//calculating the histogram
            for( int h = 0; h < histSize2-1; h++ )
            {
                float binVal = hist2.at<float>(h);
                //cout<<" "<<binVal;
                counter = counter + binVal;
            }
            counter = counter -(hist2.at<float>(3)/2); //finding the amount pixels in the cluster (last cluster plus half of the third)
            // Show the calculated histogram in command window
            
            int aim = 0;//threshold to consider per image
            int total2 = 0; //total number of pixel intensities
            double average = 0; //average pixel intensity
            int total_pixels = 0; //total pixels
            double total;
            total = gray.rows * gray.cols;
            for( int h = 0; h < histSize; h++ )
            {
                float binVal = hist.at<float>(h);
                //cout<<" "<<binVal;
                total2 = total2 + binVal; //finding thr total number of pixels
                average = average + binVal*h; //average intensity value
                total_pixels = total_pixels + binVal; //another total
                if (total2 < counter)
                    aim = h;//if intensity is within cluster
            }
            int count = 179;
            for( int h = 179; h < 240; h++ )
            {
                float binVal = hist.at<float>(h);
                //cout<<"\n "<<binVal<<"...."<<count<<endl;
                count++;
            }
            
            average = average/total_pixels;//final calculation for the average
            
            
            // Mean pixel intensity compensation per image
            if ((average != 138) && (average < 138))
                aim = aim - (138 - average);
            else if ((average != 138) && (average > 138))
                aim = aim + (average - 138);
            
            
            double aim2 = aim + 56;//57;//52; //upper threshold
            //
            aim = aim + 16;//15; //adjusting threshold
            Erosion(0, 0); //erosion for false positive reduction
            //
            threshold(grayerode, showThresh, aim,255,3);//179, 255,3);
            threshold(showThresh, showThresh, aim2, 255,4);
            
            imshow( "Gray", showThresh );
            /*if (j == 23)
             imwrite("/Users/John/Desktop/ThreshImage.png", showThresh);*/
            
            //createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
            Mat result = thresh_callback( 0, 0 );
            
            if (ct < 9)
                ss<<name<<"0"<<(ct += 1)<<type; //joining name and number.
            else
                ss<<name<<(ct += 1)<<type; //joining name and number.
            //ss<<name<<(ct += 1)<<type; //joining name and number.
            
            string filename = ss.str();
            ss.str("");
            imshow("result", gray);
            imwrite(filename, result); // saving image
            
            //waitKey(0);

            //check if the contour was drawn, if true then save images to folder
            if(drawnContour == 1){
            	if(ct < 9){
            		ss2<<name2<<"0"<<ct<<type2;
            		ss3<<name3<<"0"<<ct<<type3;
            	}
            	else{
            		ss2<<name2<<ct<<type2;
            		ss3<<name3<<ct<<type3;
            	}
            	string filename2 = ss2.str();
            	string filename3 = ss3.str();
            	ss2.str("");
            	ss3.str("");
            	drawnContour = 0;
            	imwrite(filename2, gray);
            	imwrite(filename3, showThresh);
            	
            }
        }
    }
    
    if (max_Haemorrhage_Count < 4 )
    {
        ct = 0;
        cout << "\n \n ***** No Haemorrhage Detected *****"<<endl;
        for(size_t j = 0; j < filenames.size(); ++j)//looping thriugh all the files
        {
            gray=imread(filenames[j],0);// original
            if(!gray.data)
                cerr << "Problem loading image!!!" << filenames[j] << endl ;
            else
            {
                if (ct < 9)
                    ss<<name<<"0"<<(ct += 1)<<type; //joining name and number.
                else
                    ss<<name<<(ct += 1)<<type; //joining name and number.
                //ss<<name<<(count += 1)<<type; //joining name and number.
                
                string filename = ss.str();
                ss.str("");
                imwrite(filename, gray); // saving image
            }
        }
        extaAxial = 0;
        intraCranial = 0;
    }
    else
    {
        cout<<"Solidity Ratio = "<<intraCranial/Haemorrhage_Count<<endl;
        double solidity = intraCranial/Haemorrhage_Count;
        if (solidity < 0.3)//0.2)
            cout<<"***** Extra-axial Haemorrhage *****"<<endl;
        else
            cout<<"***** Intra-axial Haemorrhage *****"<<endl;
    }
}

Mat thresh_callback(int, void* ) //taking care of the contour drawing
{
	drawnContour = 0;
    /////////
    //threshold( copyOrig, copyOrig, 245, 255, 4 );
    Mat gray2 = copyOrig;
    vector<vector<Point> > original;
    vector<Vec4i> hierarchy2;
    
    
    
    /////////
    
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    
    /// Detect edges using Threshold
    threshold( showThresh, threshold_output, thresh, 254, THRESH_BINARY );
    
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    ////////////////////////*********************
    findContours( gray2, original, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );//finding contour of the original image for comparison
    vector<double> areasOrig(original.size());
    double maxAreaOrig = 0;
    int maxOrig = 0;
    for(int i = 0; i < original.size(); i++)
    {
        areasOrig[i] = contourArea(Mat(original[i]));
        if (areasOrig[i] > maxAreaOrig)
        {
            maxAreaOrig = areasOrig[i];//finding contour with largest area of original image.
            maxOrig = i;
            
        }
    }
    double OrigPerim = 0;//original largest contour perimiter
    if (original.size()> 0)
        OrigPerim = arcLength(original[maxOrig], false);
    
    ////////////////////////*********************
    
    
    
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    
    Scalar color;
    /// Draw polygonal contour + bonding rects + circles
    int maxCountour2 = 0;
    
    double maxArea2 = 0;
    vector<double> areas(contours.size());
    
    for(int i = 0; i < contours.size(); i++)
    {
        //cout<<contours[i]<<"-----"<<endl;
        areas[i] = contourArea(Mat(contours[i]));
        if (areas[i] > maxArea2)
        {
            maxArea2 = areas[i];
            maxCountour2 = i;
            
        }
    }
    
    double perimeter = 0;
    if (contours.size() > 0)
        perimeter = arcLength(contours[maxCountour2], false);
    cout <<"Image "<<image_counter<<": Area of "<<maxArea2<<"<----"<<maxAreaOrig<<"|"<<perimeter <<"<-- "<<OrigPerim<<endl;
    image_counter++;
    Mat drawing = gray;
    
    color = Scalar(100,0,254 );//pink contour colour
    cv::cvtColor(drawing,drawing,CV_GRAY2BGR); //conversion to colour to show pink contour
    //drawContours( drawing, contours, maxCountour2, color, 2, 8, vector<Vec4i>(), 0, Point() );
    if (((maxArea2 > 3788)/*3900)*/ && (perimeter < OrigPerim/1.5/*OrigPerim - 1000*/) && ((maxAreaOrig-10000) > maxArea2) ) || (((maxArea2 > 1000 /*3055*/) && maxArea2 < 15000) && (perimeter < 1000) && ((maxAreaOrig-10000) > maxArea2) && (perimeter < OrigPerim - 1500)))
    {
    	drawnContour = 1;
    	cout <<"!!CONTOUR DRAWN"<<endl;
        drawContours( drawing, contours, maxCountour2, color, 2, 8, vector<Vec4i>(), 0, Point() ); //if conditions are satisfied, draw the contour
        false_Positive_Count += 1;
        ////////////
        vector<vector<Point> > hull2(contours.size());
        convexHull(contours[maxCountour2], hull2[maxCountour2],false);
        intraCranial += maxArea2/contourArea(hull2[maxCountour2]);
        Haemorrhage_Count +=1;
        ////////////
        if (false_Positive_Count > max_Haemorrhage_Count)//checking the amount consecutive detections
            max_Haemorrhage_Count = false_Positive_Count;
    }
    else
        false_Positive_Count = 0;
    
    
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    //waitKey(0);
    return drawing;
}

void Erosion_Haem( int, void* ) //Erosion for skull
{
    int erosion_type;
    if( erosion_elem_Haem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem_Haem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem_Haem == 2) { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    /// Apply the erosion operation
    erode( grayerode, grayerode, element );
}


