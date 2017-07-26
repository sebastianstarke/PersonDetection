/** main_ground_based_people_detection.cpp
 *
 * uses PCL1.7 to find person-sized clusters in an incoming 
 * point-cloud from the Kinect/Xtion sensor. The clusters
 * are then fed into a pre-trained SVM for classification
 * as being a person or not.
 * The vector of detected persons is published to the /persons
 * topic, and visualization_markers (potential and true persons) are published to /personMarker
 *
 * (C) 2014 Sebastian Starke
 * (C) 2014 Norman Hendrich
 *
 * 05.02.2015 - add parameters: enable, ground_frame, loop_duration
 * 31.01.2015 - rename into person_detection and person_detection_msgs
 * 17.12.2014 - split into person_recognition and person_recognition_msgs
 * 18.12.2014 - publish confidence value for use in find_user_exekutor
 * 18.12.2014 - add personDetecion/enable topic and logic
 * 08.01.2015 - added camera_optical_frame and base_footprint transformations (TODO: Map Transformation?)
 * 08.01.2015 - enhanced person tracking when the robot is moving
 * 
 * TODO: fix transformation from xtion_camera_rgb_optical_frame to /world or /map?
 */

#include <thread>

#include <ros/ros.h>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/Point.h"
#include <person_detection_msgs/Person.h>
#include <person_detection_msgs/PersonVector.h>

#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Header.h"
#include <tf/transform_listener.h>
#include <stdint.h>

#include <pcl-1.7/pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl-1.7/pcl/console/parse.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/visualization/pcl_visualizer.h>    
#include <pcl-1.7/pcl/io/openni_grabber.h>
#include <pcl-1.7/pcl/sample_consensus/sac_model_plane.h>
#include <pcl-1.7/pcl/people/ground_based_people_detection_app.h>
#include <pcl-1.7/pcl/ModelCoefficients.h>
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/sample_consensus/method_types.h>
#include <pcl-1.7/pcl/sample_consensus/model_types.h>
#include <pcl-1.7/pcl/segmentation/sac_segmentation.h>
#include <pcl-1.7/pcl/filters/passthrough.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
 
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <face_detection_msgs/FaceVector.h>

//ROS
ros::NodeHandle *nnhPtr;
ros::Subscriber subPointCloud;
ros::Subscriber enableSubscriber;

ros::Publisher truePersonsMarkerArray;
ros::Publisher personsMarkerArray;
ros::Publisher faceMatchesMarkerArray;

ros::Publisher pubPersons;
ros::Publisher pubFaces;
ros::Publisher pubConfidences;

//PointClouds
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
PointCloudT::Ptr cloud (new PointCloudT); //PC for PPL Detection
pcl::PCLPointCloud2 pcl_pc2;//Tmp Conversion PC
Eigen::VectorXf ground_coeffs;

//Classifier
pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
pcl::people::PersonClassifier<pcl::RGB> person_classifier;

cv::CascadeClassifier haar_cascade_frontalface_default;
cv::CascadeClassifier haar_cascade_mouth;

//Structs
typedef struct Face {
	cv::Rect boundingRect;
	int features;
	double count;
	bool tracked;
} Face;
typedef struct PotentialPerson {
        Eigen::Vector3f centroid;
        int count;
        bool isTruePerson;
        float confidence;
} PotentialPerson;

//Persons
std::vector<PotentialPerson> persons;
std::vector<Face> faces;
cv::Mat frame;

//Transform of frames
tf::TransformListener* listener;
std::string camera_optical_frame;
std::string ground_frame;

// Parameters: //
std::string svm_filename;
float minHeight;
float maxHeight;
int minPriorPositions;
int maxPriorPositions;
float centroidMovementDistanceThreshold;
float centroidPotentialAreaThreshold;
float minConfidence;
float relaxedConfidence;
float voxel_size;
Eigen::Matrix3f rgb_intrinsics_matrix;

int maxCount = 6;
int minCount = 3;
int minFeatures = 1;

bool detectFaces = true;
bool detectFeatures = true;

bool enableDetection = true; // enable or disable the recognition loop
double loopDuration = 0.0;
//double loopDuration = 0.1; // corresponds to 10Hz

//Functions
void trackClusterCandidates(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates);
void personDetection();
void faceDetection();
int countFeatures(cv::Mat &frame, Face &face);
bool checkIntersection(cv::Rect &rect1, cv::Rect &rect2);
cv::Point getCenter(cv::Rect &rect);

void publishConfidences(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates);
void publishPersonMarkerArray(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates);
void publishFaceMatchesMarkerArray(std::vector<Eigen::Vector3f> &clusterCentroids);
void publishTruePersonMarkerArray();

void publishFaces();
void publishPersons();

//MUTEX
bool personDetectionRunning = false;
bool faceDetectionRunning = false;

///////////////////////////////////////////////
//CODE
///////////////////////////////////////////////
void enableDetectionCallback( const std_msgs::Bool::ConstPtr& msg ) {
  enableDetection = msg->data;
  ROS_INFO( "-X- enableCallback: person detection is now enabled %d", enableDetection );
  nnhPtr->setParam( "enable", (msg->data ? 1 : 0) ); // maps to <nodename>/enable on param server
}

float distOfVectors(Eigen::Vector3f &vec1, Eigen::Vector3f &vec2)
{
  return sqrt(pow(vec2.x()-vec1.x(),2)+pow(vec2.y()-vec1.y(),2)+pow(vec2.z()-vec1.z(),2));
}

bool closeCentroidIsPotentialPerson(pcl::people::PersonCluster<PointT> &clusterCandidate)
{
  bool isPotentialPerson = false;

  Eigen::Vector3f newPersonCentroid = clusterCandidate.getCenter();
  float newPersonConfidence = clusterCandidate.getPersonConfidence();
  
  //Find closest personCentroid
  int posMinDist=-1;
  float minDist=100;
  int i=0;
  for(std::vector<PotentialPerson>::iterator it = persons.begin(); it != persons.end(); ++it) {
    if(distOfVectors(newPersonCentroid,persons.at(i).centroid)<minDist && 
       distOfVectors(newPersonCentroid,persons.at(i).centroid)<centroidPotentialAreaThreshold && 
       newPersonConfidence>relaxedConfidence)
    {
      posMinDist = i;
    }
    i++;
  }
  
  if(posMinDist!=-1) {
    isPotentialPerson = persons.at(posMinDist).count>minPriorPositions;
  }
  
  return isPotentialPerson;
}

int getCountOfVisibleTruePersons()
{
  int count=0;
    for(unsigned int i=0; i<persons.size(); i++)
    {
      if(persons.at(i).isTruePerson)
      {
	count++;
      }
    }
  
  return count;
}

float getConfidenceOfPersonCentroid(int k)
{
  float confidenceInPercent = round(100 - floorf(persons.at(k).confidence*(-62.5)*100 + 0.5)/100);
  
  if(confidenceInPercent>100)
  {
    confidenceInPercent=100;
  }
  else if(confidenceInPercent<0)
  {
    confidenceInPercent=0;
  }
  return confidenceInPercent;
}

float getDistanceOfPersonCentroid(int k)
{
  return floorf((sqrt(pow(persons.at(k).centroid.x(),2)+pow(persons.at(k).centroid.y(),2)+pow(persons.at(k).centroid.z(),2)))*100 + 0.5)/100;
}

float getHeightOfPersonCentroid(int k)
{
    //return floorf((persons.at(k).centroid.z())*100 + 15.0 + 0.5)/100; 
    return floorf((-persons.at(k).centroid.y())*100 + 148 + 0.5)/100;
    //Note that +15.0 is a small "fix" here since the camera was not perfectly calibrated. Using a perfectly calibrated camera, remove the +15.0!!!
}

void trackClusterCandidates(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates)
{
  std::vector<Eigen::Vector3f> newPersonCentroids;
  std::vector<float> newPersonConfidences;
  
  int i, j;
  
  //Filter clusterCandidates with a confidence > minConfidence or those which are very close to personClusters with a confidence > relaxedConfidence
  i=0;
  for(std::vector<pcl::people::PersonCluster<PointT> >::iterator iIt = clusterCandidates.begin(); iIt != clusterCandidates.end(); ++iIt)
  {
    //potential person aufnehmen wenn c>=c_min und dist>dist_min im movementthreshholdsinne oder wenn true person innerhalb von dist_potentialarea
    if(iIt->getPersonConfidence() > minConfidence || closeCentroidIsPotentialPerson(clusterCandidates.at(i)))
    {
     newPersonCentroids.push_back(clusterCandidates.at(i).getCenter());
     newPersonConfidences.push_back(clusterCandidates.at(i).getPersonConfidence());
    }
    i++;
  }
  
  //For each existing personCentroid, compute all potential newPersonCentroids
  i=0;
  for(unsigned int iIt=0; iIt<persons.size(); iIt++)
  { 
    //Find closest newPersonCentroid with respect to the centroidDistanceThreshold
    int posMinDist=-1;
    float minDist=100;
    j=0;
    for(std::vector<Eigen::Vector3f>::iterator jIt = newPersonCentroids.begin(); jIt != newPersonCentroids.end(); ++jIt)
    {
      if(distOfVectors(persons.at(i).centroid,newPersonCentroids.at(j))<minDist && 
	 distOfVectors(persons.at(i).centroid,newPersonCentroids.at(j))<centroidMovementDistanceThreshold)
      {
	posMinDist = j;
      }
      j++;
    }
    
    //If the newPersonCentroid is reliable, overwrite the initial personCentroid and update its count by +1
    if(posMinDist!=-1)
    {
      persons.at(i).centroid = newPersonCentroids.at(posMinDist);
      persons.at(i).confidence = newPersonConfidences.at(posMinDist);
      newPersonCentroids.erase(newPersonCentroids.begin()+posMinDist);
      
      if(persons.at(i).count < maxPriorPositions)
      {
	persons.at(i).count++;
      }
      if(persons.at(i).count >= minPriorPositions)
      {
	persons.at(i).isTruePerson = true;
      }
    }
    else //Update count by -1 and remove if the count equals 0 afterwards
    {
      persons.at(i).isTruePerson = false;
      
      if(persons.at(i).count < minPriorPositions)
      {
	persons.at(i).count=0;
      }
      else
      {
	persons.at(i).count--;
      }

      if(persons.at(i).count <= 0)
      {
	persons.erase(persons.begin()+i);
	
	i--;
      }
    }
    
    i++;
  }
  
  //Add all remaining newPersonCentroids to personCentroids to keep track of potential persons
  i=0;
  for(std::vector<Eigen::Vector3f>::iterator iIt = newPersonCentroids.begin(); iIt != newPersonCentroids.end(); ++iIt)
  {
    PotentialPerson person;
    person.centroid = newPersonCentroids.at(i);
    person.count = 1;
    person.isTruePerson = false;
    person.confidence = newPersonConfidences.at(i);

    persons.push_back(person);

    i++;
  }
}

void project3Dto2D()
{
   for(int i=0; i<persons.size(); i++)
   {
	if(persons.at(i).isTruePerson)
	{
	//Static intrinsic camera parameters, subscribe to camera_info instead!
        float f = 525.0;
    	float cx = 319.5;
    	float cy = 239.5;

	int u = f*persons.at(i).centroid.x()/persons.at(i).centroid.z() + cx;
	int v = f*persons.at(i).centroid.y()/persons.at(i).centroid.z() + cy;

	cout << "Person Image Coordinates: u=" << u << " v=" << v << endl << endl;
	}
   }
}

bool pointIsInsideBoundingRect(float x, float y, float z, float rectU, float rectV, float rectWidth, float rectHeight)
{
	bool contains = false;

	float f = 525.0;
    	float cx = 319.5;
    	float cy = 239.5;

	float u = f*x/z + cx;
	float v = f*y/z + cy;

	if(u >= rectU && u<= rectU+rectWidth && v >= rectV && v <= rectV+rectHeight)
	{
	contains = true;	
	}

	return contains;
}

void publishPersons()
{
    person_detection_msgs::PersonVector personVector;

    for(unsigned int i=0; i<persons.size(); i++)
    {
      if(persons.at(i).isTruePerson)
      {
        person_detection_msgs::Person person;

        person.position_x = persons.at(i).centroid.x();
        person.position_y = persons.at(i).centroid.y();
        person.position_z = persons.at(i).centroid.z();

        person.confidence = getConfidenceOfPersonCentroid(i);

        personVector.persons.push_back(person);
      }
    }

    personVector.frame_id = ground_frame;

    pubPersons.publish(personVector);    
}

PointT transformPoint(PointT point, std::string originFrame, std::string targetFrame)
{
    geometry_msgs::PointStamped pOrigin, pTarget;
    pOrigin.header.frame_id = originFrame;
    pOrigin.header.stamp = ros::Time();
    pOrigin.point.x = point.x;
    pOrigin.point.y = point.y;
    pOrigin.point.z = point.z;

    try{
       listener->transformPoint(targetFrame,pOrigin,pTarget);
    }
    catch (tf::ExtrapolationException ex){
        ROS_INFO("%s",ex.what());
    }
    catch (tf::LookupException ex) {
       ROS_INFO("%s",ex.what());
    }
    catch (tf::ConnectivityException ex2) {
       ROS_INFO("%s",ex2.what());
    }
    //catch ( tf::MaxDepthException ex) {
    //   ROS_WARN("%s",ex.what());
    //}

    point.x = pTarget.point.x; point.y = pTarget.point.y; point.z = pTarget.point.z;

    return point;
}

Eigen::Vector3f transformPoint(Eigen::Vector3f point, std::string originFrame, std::string targetFrame)
{
    geometry_msgs::PointStamped pOrigin, pTarget;
    pOrigin.header.frame_id = originFrame;
    pOrigin.header.stamp = ros::Time();
    pOrigin.point.x = point.x();
    pOrigin.point.y = point.y();
    pOrigin.point.z = point.z();

    try{
       listener->transformPoint(targetFrame,pOrigin,pTarget);
    }
    catch (tf::ExtrapolationException ex){
       ROS_WARN("%s",ex.what());
    }
    catch (tf::LookupException ex) {
       ROS_WARN("%s",ex.what());
    }
    catch (tf::ConnectivityException ex) {
       ROS_WARN("%s",ex.what());
    }
    //catch ( tf::MaxDepthException ex) {
    //   ROS_WARN("%s",ex.what());
    //}

    point.x() = pTarget.point.x; point.y() = pTarget.point.y; point.z() = pTarget.point.z;

    return point;
}

void personDetection()
{
    nnhPtr->getParamCached( "enable", enableDetection );
    nnhPtr->getParamCached( "ground_frame", ground_frame );
    nnhPtr->getParamCached( "loop_duration", loopDuration );
    ROS_DEBUG( "enable is now %d frame is '%s' loop duration is %6.4f", 
                enableDetection, ground_frame.c_str(), loopDuration );

    if (!enableDetection) { // save CPU cycles if not enabled
       usleep( 200*1000 );
    }
    else
    {
/*
      // Ground plane estimation:
      PointT p1, p2, p3;
      p1.x = -1.0; p1.y = 1.68; p1.z = 0.0;
      p2.x = 1.0; p2.y = 1.68; p2.z = 0.0;
      p3.x = 0.0; p3.y = 1.68; p3.z = 1.0;

      PointCloudT::Ptr groundplane_points_3d (new PointCloudT);
      groundplane_points_3d->push_back(p1);
      groundplane_points_3d->push_back(p2);
      groundplane_points_3d->push_back(p3);
      std::vector<int> groundplane_points_indices;
      for (unsigned int i = 0; i < groundplane_points_3d->points.size(); i++)
      	groundplane_points_indices.push_back(i);
      pcl::SampleConsensusModelPlane<PointT> model_plane(groundplane_points_3d);
      model_plane.computeModelCoefficients(groundplane_points_indices,ground_coeffs);
*/
 
/*
      // Ground plane estimation:
      PointT p1, p2, p3;
      p1.x = 1.0; p1.y = 0.0; p1.z = 0.0;
      p2.x = 0.0; p2.y = 1.0; p2.z = 0.0;
      p3.x = 0.0; p3.y = -1.0; p3.z = 0.0;
      p1 = transformPoint(p1,ground_frame,camera_optical_frame);
      p2 = transformPoint(p2,ground_frame,camera_optical_frame);
      p3 = transformPoint(p3,ground_frame,camera_optical_frame);

      PointCloudT::Ptr groundplane_points_3d (new PointCloudT);
      groundplane_points_3d->push_back(p1);
      groundplane_points_3d->push_back(p2);
      groundplane_points_3d->push_back(p3);
      std::vector<int> groundplane_points_indices;
      for (unsigned int i = 0; i < groundplane_points_3d->points.size(); i++)
      	groundplane_points_indices.push_back(i);
      pcl::SampleConsensusModelPlane<PointT> model_plane(groundplane_points_3d);
      model_plane.computeModelCoefficients(groundplane_points_indices,ground_coeffs);
 
      // Transform from Base-Frame to Optical-Frame
      for(int i=0; i<persons.size(); i++)
      {
      	persons.at(i).centroid = transformPoint(persons.at(i).centroid,ground_frame,camera_optical_frame);
      }
*/
      // Perform people detection on the new cloud:
      std::vector<pcl::people::PersonCluster<PointT> > clusterCandidates;   // vector containing persons clusterCandidates
      people_detector.setInputCloud(cloud);
      //people_detector.setGround(ground_coeffs);                             // set floor coefficients
      people_detector.compute(clusterCandidates);                           // perform people detection by finding clustersCandidates


      //Publish Confidences of Clusters
      publishConfidences(clusterCandidates);
      //Publish Persons detected by Basic People Detection
      publishPersonMarkerArray(clusterCandidates);
      //Publish Matches Faces on Cluster Candidates
      std::vector<Eigen::Vector3f> clusterCentroids;
      for(int i=0; i<clusterCandidates.size(); i++) {
	clusterCentroids.push_back(clusterCandidates.at(i).getCenter());
      }
      publishFaceMatchesMarkerArray(clusterCentroids);


//MATCHING
      for(int i=0; i<clusterCandidates.size(); i++)
      {
	Eigen::Vector3f centroid = clusterCandidates.at(i).getCenter();
	for(int j=0; j<faces.size(); j++)
	{
		if(pointIsInsideBoundingRect(centroid.x(), centroid.y(), centroid.z(), faces.at(j).boundingRect.x, faces.at(j).boundingRect.y, faces.at(j).boundingRect.width, faces.at(j).boundingRect.height))
		{
			clusterCandidates.at(i).setPersonConfidence(clusterCandidates.at(i).getPersonConfidence() + 1.0f);
			break;
		}
	}
      }

      trackClusterCandidates(clusterCandidates);                                   // compute additional computation and tracking steps
/*
      // Transform from Optical-Frame to Base-Frame
      for(int i=0; i<persons.size(); i++)
      {
      	persons.at(i).centroid = transformPoint(persons.at(i).centroid,camera_optical_frame,ground_frame);
      }
*/
      //Publish True Persons Fusioned with Filtered Faces
      publishTruePersonMarkerArray();
    }

   personDetectionRunning = false;

   publishPersons();
   ros::spinOnce();
}

void faceDetection()
{
    cv::Mat processingFrame;
    cvtColor( frame, processingFrame, CV_BGR2GRAY );
    //equalizeHist( processingFrame, processingFrame );

    //Reset Faces
    for(unsigned i=0; i<faces.size(); i++)
    {
      faces.at(i).tracked = false;
    }

    //Detect Faces
    std::vector<cv::Rect> faceCandidates;
    haar_cascade_frontalface_default.detectMultiScale(processingFrame, faceCandidates, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    //Merge overlapping true faces with new faces
    for(unsigned i=0; i<faces.size(); i++)
    {
        for(unsigned j=0; j<faceCandidates.size(); j++)
        {
            bool intersection = checkIntersection(faces.at(i).boundingRect,faceCandidates.at(j));
            double relativeSize = double(faces.at(i).boundingRect.size().area())/double(faceCandidates.at(j).size().area());
            if(intersection && relativeSize >= 0.8 && relativeSize <= 1.2)
            {
                faces.at(i).tracked = true;
                faces.at(i).boundingRect = faceCandidates.at(j);
                faceCandidates.erase(faceCandidates.begin()+j);
                j--;
            }
        }
    }

    //Add remaining faceCandidates to faces
    for(unsigned i=0; i<faceCandidates.size(); i++)
    {
        Face face;
        face.boundingRect = faceCandidates.at(i);
        face.count = 0;
        face.features = 0;
        face.tracked = true;
        faces.push_back(face);
    }

/*
    //Count features within frame
    for(unsigned i=0; i<faces.size(); i++)
    {
        faces.at(i).features = countFeatures(processingFrame,faces.at(i));
    }
*/

    //Update Faces
    for(unsigned i=0; i<faces.size(); i++)
    {
        if(faces.at(i).tracked)
        {
            faces.at(i).count++;
        }
        else
        {
            faces.at(i).count--;
        }

        if(!faces.at(i).tracked && faces.at(i).count < minCount)
        {
            faces.erase(faces.begin()+i);
            i--;
        }
    }

   //Correct counts
    for(unsigned i=0; i<faces.size(); i++)
    {
        if(faces.at(i).count > maxCount)
        {
            faces.at(i).count = maxCount;
        }
    }

    faceDetectionRunning = false;

    publishFaces();
    ros::spinOnce();
}
 
int countFeatures(cv::Mat &frame, Face &face)
{
  cv::Mat faceROI = frame(face.boundingRect);
  std::vector<cv::Rect> newFeatures;
  if(detectFeatures)
  {
    haar_cascade_mouth.detectMultiScale(faceROI, newFeatures);
  }
  return newFeatures.size();
}
 
bool checkIntersection(cv::Rect &rect1, cv::Rect &rect2)
{
  return (abs(rect1.x - rect2.x) * 2 < (rect1.width + rect2.width)) && (abs(rect1.y - rect2.y) * 2 < (rect1.height + rect2.height));
}

cv::Point getCenter(cv::Rect &rect)
{
    return cv::Point(rect.x + rect.width/2.0,rect.y + rect.height/2.0);
}

void publishFaces()
{
    face_detection_msgs::FaceVector faceVector;

    for(unsigned i=0; i<faces.size(); i++)
    {
        if(faces.at(i).count >= minCount)
        {
            face_detection_msgs::Face face;
            face.position_x = faces.at(i).boundingRect.x;
            face.position_y = faces.at(i).boundingRect.y;
            face.width = faces.at(i).boundingRect.width;
            face.height = faces.at(i).boundingRect.height;
            faceVector.faces.push_back(face);
        }
    }

    pubFaces.publish(faceVector);
}

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  //ROS_WARN("Cloud received!");
  if(!personDetectionRunning) {
	  personDetectionRunning = true;
	  ROS_WARN("Processing Cloud...");
	  pcl_conversions::toPCL(*msg, pcl_pc2);
	  pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

	  std::thread cloud (personDetection);
	  cloud.detach();
   }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
   //ROS_WARN("Image received!");
   if(!faceDetectionRunning) {
	   faceDetectionRunning = true;
	   ROS_WARN("Processing Image...");
	   cv_bridge::CvImagePtr cv_ptr;
	   try
	   {
	     cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	   }
	   catch (cv_bridge::Exception& e)
	   {
	       ROS_ERROR("cv_bridge exception: %s", e.what());
	       return;
	   }

	   frame = cv_ptr->image;


	   std::thread image (faceDetection);
	   image.detach();
   }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "personDetection", 1); // no SigIntHandler
  ros::NodeHandle nh; 
  ros::NodeHandle nnh( "~" ); nnhPtr = &nnh;

  image_transport::ImageTransport it_(nh);
  image_transport::Subscriber image_sub_ = it_.subscribe("/camera/rgb/image_color", 1, &imageCallback);

    std::string fn_haar_frontalface_default = "/opt/ros/hydro/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
    std::string haarcascade_mcs_mouth = "/opt/ros/hydro/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";

    haar_cascade_frontalface_default.load(fn_haar_frontalface_default);
    haar_cascade_mouth.load(haarcascade_mcs_mouth);
  
  if(argc > 0)
  {	
    subPointCloud = nh.subscribe(argv[1], 1, cloudCallback);
    svm_filename = argv[2];
    minHeight = atof(argv[3]);
    maxHeight = atof(argv[4]);
    minPriorPositions = atoi(argv[5]);
    maxPriorPositions = atoi(argv[6]);
    centroidMovementDistanceThreshold = atof(argv[7]);
    centroidPotentialAreaThreshold = atof(argv[8]);
    minConfidence = atof(argv[9]);
    relaxedConfidence = atof(argv[10]);
    camera_optical_frame = argv[11];
    ground_frame = argv[12];

    voxel_size = 0.06;
    rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0;
  }

  // Create classifier for people detection:
  person_classifier.loadSVMFromFile(svm_filename);                       // load trained SVM

  // People detection app initialization:
  people_detector.setVoxelSize(voxel_size);                              // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);                  // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                      // set person classifier
  people_detector.setHeightLimits(minHeight, maxHeight);               // set person classifier

  ground_coeffs.resize(4);
  
  nnh.setParam( "enable", true );
  nnh.setParam( "ground_frame", ground_frame );
  nnh.setParam( "loop_duration", loopDuration );
  ROS_WARN( "setting parameters: 'enable'=%d 'ground_frame'='%s'", true, ground_frame.c_str() );
  ROS_WARN( "loop duration is %6.4f", loopDuration );

  enableSubscriber = nh.subscribe( "personDetection/enable", 1, enableDetectionCallback );

  faceMatchesMarkerArray = nh.advertise<visualization_msgs::MarkerArray>( "faceMatches",1);

  truePersonsMarkerArray = nh.advertise<visualization_msgs::MarkerArray>( "truePersonMarkers",1);
  personsMarkerArray = nh.advertise<visualization_msgs::MarkerArray>( "personMarkers", 1 );

  pubFaces = nh.advertise<face_detection_msgs::FaceVector>( "faces", 1);
  pubPersons = nh.advertise<person_detection_msgs::PersonVector>( "persons", 1);  

  pubConfidences = nh.advertise<visualization_msgs::MarkerArray>( "confidences", 1);

  listener = new tf::TransformListener();

      // Ground plane estimation:
      PointT p1, p2, p3;
      p1.x = -1.0; p1.y = 1.68; p1.z = 0.0;
      p2.x = 1.0; p2.y = 1.68; p2.z = 0.0;
      p3.x = 0.0; p3.y = 1.68; p3.z = 1.0;

      PointCloudT::Ptr groundplane_points_3d (new PointCloudT);
      groundplane_points_3d->push_back(p1);
      groundplane_points_3d->push_back(p2);
      groundplane_points_3d->push_back(p3);
      std::vector<int> groundplane_points_indices;
      for (unsigned int i = 0; i < groundplane_points_3d->points.size(); i++)
      	groundplane_points_indices.push_back(i);
      pcl::SampleConsensusModelPlane<PointT> model_plane(groundplane_points_3d);
      model_plane.computeModelCoefficients(groundplane_points_indices,ground_coeffs);
      people_detector.setGround(ground_coeffs);                             // set floor coefficients

  for(;;)
  {
      ros::spinOnce();
  }

  return 0;
}

//VISUALIZATION STUFF
/////////////////////

//TODO: Auch die clustercentroids reingeben von potentialfaces struct statt nur clustercandidates
void publishFaceMatchesMarkerArray(std::vector<Eigen::Vector3f> &clusterCentroids) {
      visualization_msgs::MarkerArray markerArray;
      int k=0;
      for(int i=0; i<clusterCentroids.size(); i++)
      {
	Eigen::Vector3f centroid = clusterCentroids.at(i);
	for(int j=0; j<faces.size(); j++)
	{
		if(pointIsInsideBoundingRect(centroid.x(), centroid.y(), centroid.z(), faces.at(j).boundingRect.x, faces.at(j).boundingRect.y, faces.at(j).boundingRect.width, faces.at(j).boundingRect.height))
		{
			visualization_msgs::Marker marker;
			marker.header.frame_id = camera_optical_frame;
			marker.header.stamp = ros::Time();
			marker.ns = "faces";
			marker.id = k+7000;
			marker.type = visualization_msgs::Marker::LINE_STRIP;
			marker.action = visualization_msgs::Marker::ADD;
			//marker.pose.position.x = (double) centroid.x();
			//marker.pose.position.y = (double) centroid.y();
			//marker.pose.position.z = (double) centroid.z();

			marker.pose.orientation.x = 0.0;
			marker.pose.orientation.y = 0.0;
			marker.pose.orientation.z = 0.0;
			marker.pose.orientation.w = 1.0;

			marker.points.resize(5);
                        marker.points[0].x = centroid.x() - 0.15;
                        marker.points[1].x = centroid.x() - 0.15;
                        marker.points[2].x = centroid.x() + 0.15;
                        marker.points[3].x = centroid.x() + 0.15;
                        marker.points[4].x = centroid.x() - 0.15;

                        marker.points[0].y = centroid.y() - 0.15;
                        marker.points[1].y = centroid.y() + 0.15;
                        marker.points[2].y = centroid.y() + 0.15;
                        marker.points[3].y = centroid.y() - 0.15;
                        marker.points[4].y = centroid.y() - 0.15;

                        marker.points[0].z = centroid.z();
                        marker.points[1].z = centroid.z();
                        marker.points[2].z = centroid.z();
                        marker.points[3].z = centroid.z();
                        marker.points[4].z = centroid.z();

			marker.scale.x = 0.01;
			//marker.scale.y = 0.5;
			//marker.scale.z = 0.5;
			marker.color.a = 1.0;
          		marker.color.r = 0.0;
          		marker.color.g = 0.0;
         		marker.color.b = 1.0;
        		marker.lifetime = ros::Duration(0.25);
        		markerArray.markers.push_back(marker);
			k++;

			break;
		}
	}
      }
      faceMatchesMarkerArray.publish(markerArray);
}

void publishTruePersonMarkerArray() {
      visualization_msgs::MarkerArray marker_array;
      for(unsigned int k=0; k<persons.size(); k++)
      {
	if(persons.at(k).isTruePerson) {
		visualization_msgs::Marker marker;
		marker.header.frame_id = camera_optical_frame;
		marker.header.stamp = ros::Time();
		marker.ns = "persons";
		marker.id = 100+k;
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = (double) persons.at(k).centroid.x();
		marker.pose.position.y = (double) persons.at(k).centroid.y();
		marker.pose.position.z = (double) persons.at(k).centroid.z();
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.25;
		marker.scale.y = 0.25;
		marker.scale.z = 0.25;

		visualization_msgs::Marker markerText;
		markerText.header.frame_id = camera_optical_frame;
		markerText.header.stamp = ros::Time();
		markerText.ns = "personsInfo";
		markerText.id = 200+k;
		markerText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
		markerText.action = visualization_msgs::Marker::ADD;
		markerText.pose.position.x = (double) persons.at(k).centroid.x();
		markerText.pose.position.y = (double) persons.at(k).centroid.y()+0.5;
		markerText.pose.position.z = (double) persons.at(k).centroid.z()-0.5;
		markerText.pose.orientation.x = 0.0;
		markerText.pose.orientation.y = 0.0;
		markerText.pose.orientation.z = 0.0;
		markerText.pose.orientation.w = 1.0;
		markerText.scale.z = 0.15;
	
//		if(persons.at(k).isTruePerson)
//		{
		  marker.color.a = 0.5;
		  marker.color.r = 1.0 - (double)persons.at(k).count/(double)maxPriorPositions;
		  marker.color.g = 0.0 + (double)persons.at(k).count/(double)maxPriorPositions;
		  marker.color.b = 0.0;

		  markerText.color.a = 1.0;
		  markerText.color.r = 0.0;
		  markerText.color.g = 1.0;
		  markerText.color.b = 1.0;
		  markerText.text=
		  "\nConfidence: " + boost::to_string(getConfidenceOfPersonCentroid(k)) + "%" +
		  "\nDistance: " + boost::to_string(getDistanceOfPersonCentroid(k)) + "m" +
		  "\nHeight: " + boost::to_string(getHeightOfPersonCentroid(k)) + "m";
/*
		}
		else
		{
		  marker.color.a = 1.0;
		  marker.color.r = 1.0;
		  marker.color.g = 1.0;
		  marker.color.b = 1.0;

		  markerText.color.a = 1.0;
		  markerText.color.r = 1.0;
		  markerText.color.g = 0.0;
		  markerText.color.b = 0.0;
		  markerText.text="Potential Person";
		}
*/
		  
		marker.lifetime = ros::Duration(0.25);
		markerText.lifetime = ros::Duration(0.25);

		marker_array.markers.push_back(marker);
		marker_array.markers.push_back(markerText);
	}
      }
      
      truePersonsMarkerArray.publish(marker_array);
}

void publishPersonMarkerArray(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates)
{
      visualization_msgs::MarkerArray marker_array;
      int i=0;
      for(std::vector<pcl::people::PersonCluster<PointT> >::iterator iIt = clusterCandidates.begin(); iIt != clusterCandidates.end(); ++iIt)
      {
	float confidence = iIt->getPersonConfidence();
	if(confidence >= minConfidence) {
		float centroidX = iIt->getCenter().x();
		float centroidY = iIt->getCenter().y();
		float centroidZ = iIt->getCenter().z();

		visualization_msgs::Marker marker;
		marker.header.frame_id = camera_optical_frame;
		marker.header.stamp = ros::Time();
		marker.ns = "persons";
		marker.id = 0+i;
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = (double) centroidX;
		marker.pose.position.y = (double) centroidY;
		marker.pose.position.z = (double) centroidZ;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.2;
		marker.scale.y = 0.2;
		marker.scale.z = 0.2;
		marker.color.a = 1.0;
		marker.color.r = 1.0;
		marker.color.g = 1.0;
		marker.color.b = 1.0;
		  
		marker.lifetime = ros::Duration(0.25);

		marker_array.markers.push_back(marker);
		i++;
	}
      }
      
      personsMarkerArray.publish(marker_array);
}

void publishConfidences(std::vector<pcl::people::PersonCluster<PointT> > &clusterCandidates)
{
    visualization_msgs::MarkerArray markerArray;
    int i=0;
    for(std::vector<pcl::people::PersonCluster<PointT> >::iterator iIt = clusterCandidates.begin(); iIt != clusterCandidates.end(); ++iIt)
    {
        float confidence = iIt->getPersonConfidence();
        visualization_msgs::Marker markerText;
        markerText.header.frame_id = camera_optical_frame;
        markerText.header.stamp = ros::Time();
        markerText.ns = "personsConfidences";
        markerText.id = i+15000;
        markerText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerText.action = visualization_msgs::Marker::ADD;
        markerText.pose.position.x = (double) iIt->getCenter().x();
        markerText.pose.position.y = (double) iIt->getCenter().y();
        markerText.pose.position.z = (double) iIt->getCenter().z();
        markerText.pose.orientation.x = 0.0;
        markerText.pose.orientation.y = 0.0;
        markerText.pose.orientation.z = 0.0;
        markerText.pose.orientation.w = 1.0;
        markerText.scale.z = 0.15;

        markerText.color.a = 1.0;
        markerText.color.r = 1.0;
        markerText.color.g = 0.0;
        markerText.color.b = 0.0;
        char tmp[20];
        sprintf(tmp, "%lf", confidence);
        markerText.text = tmp;

        markerArray.markers.push_back(markerText);
        i++;
    }
    pubConfidences.publish(markerArray);
}
