#include <iostream>
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <glog/logging.h>
#include <iterator>
#define EPS T(0.00001)
double fx, fy, tx, ty, w3Dv2D;
using namespace pybind11::literals;
namespace py = pybind11;


struct AlignmentErrorTriangulate {
  AlignmentErrorTriangulate(double* observed_in, double* camera_extrinsic_in, double* camera_intrinsic_in): observed(observed_in), camera_extrinsic(camera_extrinsic_in),camera_intrinsic(camera_intrinsic_in) {}


  template <typename T>
  bool operator()(const T* const point, T* residuals) const {
                  
    // camera_extrinsic[0,1,2] are the angle-axis rotation.
    T p[3];
    //ceres::AngleAxisRotatePoint((T*)(camera_extrinsic), point, p);
    p[0] = camera_extrinsic[0]*point[0] + camera_extrinsic[1]*point[1] + camera_extrinsic[2]*point[2];
    p[1] = camera_extrinsic[3]*point[0] + camera_extrinsic[4]*point[1] + camera_extrinsic[5]*point[2];
    p[2] = camera_extrinsic[6]*point[0] + camera_extrinsic[7]*point[1] + camera_extrinsic[8]*point[2];
  
    // camera_extrinsic[3,4,5] are the translation.
    p[0] += T(camera_extrinsic[9]);
    p[1] += T(camera_extrinsic[10]);
    p[2] += T(camera_extrinsic[11]);
 
    // let p[2] ~= 0
    if (T(0.0)<=p[2]){
        if(p[2]<EPS){
            p[2] = EPS;
        }
    }else{
        if (p[2]>-EPS){
            p[2] = -EPS;
        }
    }
    
    // project it
    p[0] = T(camera_intrinsic[0]) * p[0] / p[2] + T(camera_intrinsic[2]);
    p[1] = T(camera_intrinsic[4]) * p[1] / p[2] + T(camera_intrinsic[5]);
    
    // reprojection error
    residuals[0] = (p[0] - T(observed[0]));
    residuals[1] = (p[1] - T(observed[1]));     

    
    //std::cout<<"p[0]="<<p[0]<<std::endl;
    //std::cout<<"p[1]="<<p[1]<<std::endl;
    //std::cout<<"observed[0]="<<observed[0]<<std::endl;
    //std::cout<<"observed[1]="<<observed[1]<<std::endl;
    //std::cout<<"residuals[0]="<<residuals[0]<<std::endl;
    //std::cout<<"residuals[1]="<<residuals[1]<<std::endl;
    //std::cout<<"camera_intrinsic[0]="<<camera_intrinsic[0]<<std::endl;
    //std::cout<<"camera_extrinsic[3]="<<camera_extrinsic[3]<<std::endl;
    //std::cout<<"--------------------------"<<std::endl;
    
    return true;
  }
  
  double* observed;
  double* camera_extrinsic;
  double* camera_intrinsic;
};

struct AlignmentErrorTrajectory {
  AlignmentErrorTrajectory() {}

  template <typename T>
  bool operator()(const T* const car_extrinsic_t_old,const T* const car_extrinsic_t,const T* const car_extrinsic_next, T* residuals) const {
           
    T old[3];
    old[0]= car_extrinsic_t_old[0] - car_extrinsic_t[0];       
    old[1]= car_extrinsic_t_old[1] - car_extrinsic_t[1];       
    old[2]= car_extrinsic_t_old[2] - car_extrinsic_t[2];       

    T next[3];
    next[0]= car_extrinsic_t[0] - car_extrinsic_next[0];       
    next[1]= car_extrinsic_t[1] - car_extrinsic_next[1];       
    next[2]= car_extrinsic_t[2] - car_extrinsic_next[2];       
    
    residuals[0] = (old[0] - next[0]);
    residuals[1] = (old[1] - next[1]);
    residuals[2] = (old[2] - next[2]);

    
    return true;
  }
  
  double* observed;
  double* camera_extrinsic;
  double* camera_intrinsic;
};


struct AlignmentErrorCar {
  AlignmentErrorCar(double* observed_in, double* camera_extrinsic_in, double* camera_intrinsic_in, double* keyp_3d_in): observed(observed_in), camera_extrinsic(camera_extrinsic_in),camera_intrinsic(camera_intrinsic_in),keyp_3d(keyp_3d_in) {}

  template <typename T>
  bool operator()(const T* const car_extrinsic,const T* const alpha, T* residuals) const {
                  
    // camera_extrinsic[0,1,2] are the angle-axis rotation.
    //std::cout<<"p[1]="<<alpha[0]<<std::endl;

    T p[3];
    T o[3];
    T temp_p[3];
    o[0] = alpha[0]*keyp_3d[0];
    o[1] = alpha[0]*keyp_3d[1];
    o[2] = alpha[0]*keyp_3d[2];
    //std::cout<<"p[0]="<<o[0]<<std::endl;
    //std::cout<<"p[1]="<<o[1]<<std::endl;
    //std::cout<<"p[2]="<<o[2]<<std::endl;
    T carRT[12];    
    ceres::AngleAxisToRotationMatrix(car_extrinsic, carRT);
    carRT[9] = car_extrinsic[3];
    carRT[10] = car_extrinsic[4];
    carRT[11] = car_extrinsic[5];    
    temp_p[0] = T(carRT[0])*o[0] + T(carRT[1])*o[1] + T(carRT[2])*o[2];
    temp_p[1] = T(carRT[3])*o[0] + T(carRT[4])*o[1] + T(carRT[5])*o[2];
    temp_p[2] = T(carRT[6])*o[0] + T(carRT[7])*o[1] + T(carRT[8])*o[2];
  

    temp_p[0] += T(car_extrinsic[3]);
    temp_p[1] += T(car_extrinsic[4]);
    temp_p[2] += T(car_extrinsic[5]);
    //std::cout<<"p[0]="<<temp_p[0]<<std::endl;
    //std::cout<<"p[1]="<<temp_p[1]<<std::endl;
    //std::cout<<"p[2]="<<temp_p[2]<<std::endl;
    
    p[0] = T(camera_extrinsic[0])*temp_p[0] + T(camera_extrinsic[1])*temp_p[1] + T(camera_extrinsic[2])*temp_p[2];
    p[1] = T(camera_extrinsic[3])*temp_p[0] + T(camera_extrinsic[4])*temp_p[1] + T(camera_extrinsic[5])*temp_p[2];
    p[2] = T(camera_extrinsic[6])*temp_p[0] + T(camera_extrinsic[7])*temp_p[1] + T(camera_extrinsic[8])*temp_p[2];
    // camera_extrinsic[3,4,5] are the translation.
    p[0] += T(camera_extrinsic[9]);
    p[1] += T(camera_extrinsic[10]);
    p[2] += T(camera_extrinsic[11]);
 
    // let p[2] ~= 0
    if (T(0.0)<=p[2]){
        if(p[2]<EPS){
            p[2] = EPS;
        }
    }else{
        if (p[2]>-EPS){
            p[2] = -EPS;
        }
    }
   
    // project it
    p[0] = T(camera_intrinsic[0]) * p[0] / p[2] + T(camera_intrinsic[2]);
    p[1] = T(camera_intrinsic[4]) * p[1] / p[2] + T(camera_intrinsic[5]);
    
    // reprojection error
    residuals[0] = (p[0] - T(observed[0]));
    residuals[1] = (p[1] - T(observed[1]));     

    
    //std::cout<<"p[0]="<<p[0]<<std::endl;
    //std::cout<<"p[1]="<<p[1]<<std::endl;    
    //std::cout<<"observed[0]="<<observed[0]<<std::endl;
    //std::cout<<"observed[1]="<<observed[1]<<std::endl;
    //std::cout<<"residuals[0]="<<residuals[0]<<std::endl;
    //std::cout<<"residuals[1]="<<residuals[1]<<std::endl;
    //std::cout<<"camera_intrinsic[0]="<<carRT[0]<<std::endl;
    //std::cout<<"camera_intrinsic[0]="<<temp_p[0]<<" "<<p[0]<<" "<<residuals[0]<<" "<<carRT[0]<<" "<<alpha[0]<<" "<<keyp_3d[0]<<" "<<camera_intrinsic[0]<<std::endl;
    //std::cout<<"camera_extrinsic[3]="<<car_extrin sic[4]<<std::endl;
    //std::cout<<"--------------------------"<<std::endl;
    
    return true;
  }
  
  double* observed;
  double* camera_extrinsic;
  double* camera_intrinsic;
  double* keyp_3d;
};


double* read_data(py::buffer x0){
	auto buf1 = x0.request();
    if (buf1.ndim != 1) {
        throw std::runtime_error("Number of dimensions of x0 must be one");
    }

    auto result = py::array_t<double>(buf1.size);
    auto buf2 = result.request();
    
    double* data = static_cast<double*>(buf2.ptr);
    double* inputs = static_cast<double*>(buf1.ptr);

    for (unsigned long i = 0; i < buf1.size; i++) {
        data[i] = inputs[i];
    }
    return data;
	}

void read_data_check(py::buffer x0,double *actual_data[]){
	auto buf1 = x0.request();
    if (buf1.ndim != 1) {
        throw std::runtime_error("Number of dimensions of x0 must be one");
    }

   
    double* inputs = static_cast<double*>(buf1.ptr);

    for (unsigned long i = 0; i < buf1.size; i++) {
        actual_data[i] = &inputs[i];
    }
	}
	
void read_data_check1(py::buffer x1,double *act_data[]){
	//std::cout<<"check";
	auto buf1 = x1.request();
    double* inputs = static_cast<double*>(buf1.ptr);

    for (unsigned long i = 0; i < buf1.size; i++) {
        act_data[i] = &inputs[i];
    }
}
py::object ba_optimize(py::function func, py::function grad, py::buffer x0,py::buffer x1,py::buffer x2,py::buffer x3,py::buffer x4,py::buffer x5,py::buffer x6,py::buffer x7) {
	auto buf0 = x0.request();
	double* data = static_cast<double*>(buf0.ptr);
	unsigned int mode = data[0];
	unsigned int nCam = data[1];
	unsigned int nPts = data[2];
	unsigned int nObs = data[3];
	  std::cout<<"Parameters: ";
	  std::cout<<"mode="<<mode<<" ";

	  std::cout<<"Meta Info: ";
	  std::cout<<"nCam="<<nCam<<" ";
	  std::cout<<"nPts="<<nPts<<" ";
	  std::cout<<"nObs="<<nObs<<"\t"<<std::endl;

  ////// construct camera parameters from camera matrix
  auto buf1 = x1.request();
  double* cameraRt = static_cast<double*>(buf1.ptr);
  double* cameraParameter = new double [12*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraPtr = cameraParameter+12*cameraID;
      double* cameraMat = cameraRt+12*cameraID;
      if (!(std::isnan(*cameraPtr))){
          cameraPtr[0] = cameraMat[0];
          cameraPtr[1] = cameraMat[1];
          cameraPtr[2] = cameraMat[2];
          cameraPtr[3] = cameraMat[3];
          cameraPtr[4] = cameraMat[4];
          cameraPtr[5] = cameraMat[5];
          cameraPtr[6] = cameraMat[6];
          cameraPtr[7] = cameraMat[7];
          cameraPtr[8] = cameraMat[8];
          cameraPtr[9] = cameraMat[9];
          cameraPtr[10] = cameraMat[10];
          cameraPtr[11] = cameraMat[11];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPtr="<<cameraPtr[0]<<" "<<cameraPtr[1]<<" "<<cameraPtr[2]<<" "<<cameraPtr[3]<<" "<<cameraPtr[4]<<" "<<cameraPtr[5]<<std::endl;
      }
  }	  

  auto buf2 = x2.request();
  double* cameraK = static_cast<double*>(buf2.ptr);
  double* cameraIntrinscis = new double [9*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraInt = cameraIntrinscis+9*cameraID;
      double* cameraMat_k = cameraK+9*cameraID;
      if (!(std::isnan(*cameraInt))){
          cameraInt[0] = cameraMat_k[0];
          cameraInt[1] = cameraMat_k[1];
          cameraInt[2] = cameraMat_k[2];
          cameraInt[3] = cameraMat_k[3];
          cameraInt[4] = cameraMat_k[4];
          cameraInt[5] = cameraMat_k[5];
          cameraInt[6] = cameraMat_k[6];
          cameraInt[7] = cameraMat_k[7];
          cameraInt[8] = cameraMat_k[8];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraInt="<<cameraInt[0]<<" "<<cameraInt[1]<<" "<<cameraInt[2]<<" "<<cameraInt[3]<<" "<<cameraInt[4]<<" "<<cameraInt[5]<<std::endl;
      }
  }	  

  auto buf3 = x3.request();
  double* location_2d = static_cast<double*>(buf3.ptr);
  double* camera_2d = new double [2*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraPtr_2d = camera_2d+2*cameraID;
      double* cameraMat_2d = location_2d+2*cameraID;
      if (!(std::isnan(*cameraPtr_2d))){
          cameraPtr_2d[0] = cameraMat_2d[0];
          cameraPtr_2d[1] = cameraMat_2d[1];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPtr_2d="<<cameraPtr_2d[0]<<" "<<cameraPtr_2d[1]<<std::endl;
      }
  }	
  
  auto buf4 = x4.request();
  double* location_3d = static_cast<double*>(buf4.ptr);	 
  double* pointCloud = new double [3*nPts];
  for(int cameraID=0; cameraID<nPts; ++cameraID){
      double* cameraPoint = pointCloud+3*cameraID;
      double* cameraMat_3d = location_3d+3*cameraID;
      if (!(std::isnan(*cameraPoint))){
          cameraPoint[0] = cameraMat_3d[0];
          cameraPoint[1] = cameraMat_3d[1];
          cameraPoint[2] = cameraMat_3d[2];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPoint="<<cameraPoint[0]<<" "<<cameraPoint[1]<<std::endl;
      }
  }	 

	
	auto buf5 = x5.request();
	double* correspondence_new = static_cast<double*>(buf5.ptr);	
    double* correspondence = new double [nObs];
    for(int cameraID=0; cameraID<nObs; ++cameraID){
		double* correspondencePtr = correspondence+cameraID;
		double* correspondenceMat = correspondence_new+cameraID;
		correspondencePtr[0] = correspondenceMat[0];
	}


    double* alpha = read_data(x6);

	  // construct camera parameters from camera matrix
	  auto buf7 = x7.request();
	  double* carRt = static_cast<double*>(buf7.ptr);
	  double* carPtr = new double [6];
      //std::cout<<"carPtr="<<alpha[0]<<" "<<carRt[0]<<" "<<carRt[1]<<" "<<carRt[2]<<" "<<carRt[3]<<" "<<carRt[4]<<" "<<carRt[9]<<std::endl;
      ceres::RotationMatrixToAngleAxis<double>(carRt, carPtr);
      carPtr[3] = carRt[9];
      carPtr[4] = carRt[10];
      carPtr[5] = carRt[11];
      //double* carRt_new = new double [12];
      //ceres::AngleAxisToRotationMatrix<double>(carPtr,carRt_new);
      //carRt_new[9] = carPtr[3];
      //carRt_new[10] = carPtr[4];
      //carRt_new[11] = carPtr[5];
      //std::cout<<"carRt_new="<<alpha[0]<<" "<<carRt_new[0]<<" "<<carRt_new[1]<<" "<<carRt_new[2]<<" "<<carRt_new[3]<<" "<<carRt_new[4]<<" "<<carRt_new[9]<<std::endl;
            

	  // output info


	  //std::cout<<"Camera Intrinsic: "<<correspondence[0];
	  //std::cout<<"fx="<<fx<<" ";
	  //std::cout<<"fy="<<fy<<" ";
	  //std::cout<<"px="<<px<<" ";
	  //std::cout<<"py="<<py<<"\t"<<std::endl;	


	  ceres::Problem problem;
	  ceres::LossFunction* loss_function = NULL; // squared loss    
	  //double* alpha = new double [1]();
	  //alpha[0] = 1.0;
	  //double* carRt = new double [6]();

	  for (unsigned int idObs=0; idObs<nObs; ++idObs){
		double* cameraPtr_ceres = cameraParameter + idObs * 12;
		double* cameraInt_ceres = cameraIntrinscis + idObs * 9;
		double* observePtr = camera_2d+2*idObs;
		int counter = (int) correspondence[idObs];
		double* pointPtr  = pointCloud + 3*counter;
        //std::cout<<"cameraPtr="<<counter<<std::endl;
		//if (idObs==0){
		//std::cout<<"cameraPtr="<<" "<<observePtr[1];
		//std::cout<<" "<<cameraPtr_ceres[0];
		//std::cout<<" "<<cameraInt_ceres[0];
		//std::cout<<" "<<pointPtr[0];
		//std::cout<<" "<<carPtr[0];
		//std::cout<<" "<<alpha[0];
		//std::cout<<" "<<counter<<std::endl;
		//}
		ceres::CostFunction* cost_function;
        //cost_function = new ceres::AutoDiffCostFunction<AlignmentErrorTriangulate, 2, 3>(new AlignmentErrorTriangulate(observePtr,cameraPtr,cameraInt));
        //problem.AddResidualBlock(cost_function,loss_function,pointPtr);
        cost_function = new ceres::AutoDiffCostFunction<AlignmentErrorCar, 2, 6,1>(new AlignmentErrorCar(observePtr,cameraPtr_ceres,cameraInt_ceres,pointPtr));
        problem.AddResidualBlock(cost_function,loss_function,carPtr,alpha);
		}
		
		 ceres::Solver::Options options;
		 options.max_num_iterations = 200;  
		 options.minimizer_progress_to_stdout = true;
		 options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;
		  ////options.ordering_type = ceres::SCHUR;
		 ceres::Solver::Summary summary;
		 ceres::Solve(options, &problem, &summary);
	    std::cout << summary.BriefReport() << "\n";
	  double* carRt_op = new double [12];
      ceres::AngleAxisToRotationMatrix<double>(carPtr,carRt_op);
	  carRt_op[9] = carPtr[3];
	  carRt_op[10] = carPtr[4];
	  carRt_op[11] = carPtr[5];
      //std::cout<<"carRt_op="<<alpha[0]<<" "<<carRt_op[0]<<" "<<carRt_op[1]<<" "<<carRt_op[2]<<" "<<carRt_op[3]<<" "<<carRt_op[4]<<" "<<carRt_op[9]<<std::endl;






    ////ceres::Solve(options, problem, data, &summary);


    
    //auto iterations = summary.iterations;
    //int nfev = 0;
    //int ngev = 0;
    //for (auto& summ: iterations) {
        //nfev += summ.line_search_function_evaluations;
        //ngev += summ.line_search_gradient_evaluations;
    //}

    auto OptimizeResult = py::module::import("scipy.optimize").attr("OptimizeResult");
    //std::string s(p, p + sizeof &carRt_op); 
    
    //std::cout<<sizeof(carRt_op)<<std::endl;
    std::stringstream ss;
    for(size_t i = 0; i < sizeof(carRt_op)+1; i++)
    {
  	if(i != 0)
    	ss << ",";
  	ss << carRt_op[i];
  }
  for(size_t j = 3; j < sizeof(carPtr)-2; j++)
    {
    	ss << ",";
  	ss << carPtr[j];
    }
     std::string s = ss.str();

    py::dict out("x"_a = pointCloud[0],
				 "y"_a = pointCloud[1],
				 "z"_a = pointCloud[2],
				 "rt00"_a = carRt_op[0],
				 "rt01"_a = carRt_op[1],
				 "rt02"_a = carRt_op[2],
				 "rt10"_a = carRt_op[3],
				 "rt11"_a = carRt_op[4],
				 "rt12"_a = carRt_op[5],
				 "rt20"_a = carRt_op[6],
				 "rt21"_a = carRt_op[7],
				 "rt22"_a = carRt_op[8],
				 "rt03"_a = carPtr[3],
				 "rt13"_a = carPtr[4],
				 "rt23"_a = carPtr[5],
				 "scale"_a = alpha[0],
				 "rt"_a = s,
                 "success"_a = summary.termination_type == ceres::CONVERGENCE ||
                               summary.termination_type == ceres::USER_SUCCESS,
                 "status"_a = (int)summary.termination_type,
                 "message"_a = summary.message,
                 "fun"_a = summary.final_cost
                 );

    return OptimizeResult(out);
    //return x0;
}

	


py::object ba_optimize_video(py::function func, py::function grad, py::buffer x0,py::buffer x1,py::buffer x2,py::buffer x3,py::buffer x4,py::buffer x5,py::buffer x6,py::buffer x7,py::buffer x8) {
	
	auto buf0 = x0.request();
	double* data = static_cast<double*>(buf0.ptr);
	unsigned int mode = data[0];
	unsigned int nCam = data[1];
	unsigned int nPts = data[2];
	unsigned int nObs = data[3];
	  std::cout<<"Parameters: ";
	  std::cout<<"mode="<<mode<<" ";

	  std::cout<<"Meta Info: ";
	  std::cout<<"nCam="<<nCam<<" ";
	  std::cout<<"nPts="<<nPts<<" ";
	  std::cout<<"nObs="<<nObs<<"\t"<<std::endl;

  auto buf1 = x1.request();
  double* cameraRt = static_cast<double*>(buf1.ptr);
  double* cameraParameter = new double [12*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraPtr = cameraParameter+12*cameraID;
      double* cameraMat = cameraRt+12*cameraID;
      if (!(std::isnan(*cameraPtr))){
          cameraPtr[0] = cameraMat[0];
          cameraPtr[1] = cameraMat[1];
          cameraPtr[2] = cameraMat[2];
          cameraPtr[3] = cameraMat[3];
          cameraPtr[4] = cameraMat[4];
          cameraPtr[5] = cameraMat[5];
          cameraPtr[6] = cameraMat[6];
          cameraPtr[7] = cameraMat[7];
          cameraPtr[8] = cameraMat[8];
          cameraPtr[9] = cameraMat[9];
          cameraPtr[10] = cameraMat[10];
          cameraPtr[11] = cameraMat[11];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPtr="<<cameraPtr[0]<<" "<<cameraPtr[1]<<" "<<cameraPtr[2]<<" "<<cameraPtr[3]<<" "<<cameraPtr[4]<<" "<<cameraPtr[5]<<std::endl;
      }
  }	  

  auto buf2 = x2.request();
  double* cameraK = static_cast<double*>(buf2.ptr);
  double* cameraIntrinscis = new double [9*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraInt = cameraIntrinscis+9*cameraID;
      double* cameraMat_k = cameraK+9*cameraID;
      if (!(std::isnan(*cameraInt))){
          cameraInt[0] = cameraMat_k[0];
          cameraInt[1] = cameraMat_k[1];
          cameraInt[2] = cameraMat_k[2];
          cameraInt[3] = cameraMat_k[3];
          cameraInt[4] = cameraMat_k[4];
          cameraInt[5] = cameraMat_k[5];
          cameraInt[6] = cameraMat_k[6];
          cameraInt[7] = cameraMat_k[7];
          cameraInt[8] = cameraMat_k[8];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraInt="<<cameraInt[0]<<" "<<cameraInt[1]<<" "<<cameraInt[2]<<" "<<cameraInt[3]<<" "<<cameraInt[4]<<" "<<cameraInt[5]<<std::endl;
      }
  }	  

  auto buf3 = x3.request();
  double* location_2d = static_cast<double*>(buf3.ptr);
  double* camera_2d = new double [2*nObs];
  for(int cameraID=0; cameraID<nObs; ++cameraID){
      double* cameraPtr_2d = camera_2d+2*cameraID;
      double* cameraMat_2d = location_2d+2*cameraID;
      if (!(std::isnan(*cameraPtr_2d))){
          cameraPtr_2d[0] = cameraMat_2d[0];
          cameraPtr_2d[1] = cameraMat_2d[1];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPtr_2d="<<cameraPtr_2d[0]<<" "<<cameraPtr_2d[1]<<std::endl;
      }
  }	
  
  auto buf4 = x4.request();
  double* location_3d = static_cast<double*>(buf4.ptr);	 
  double* pointCloud = new double [3*nPts];
  for(int cameraID=0; cameraID<nPts; ++cameraID){
      double* cameraPoint = pointCloud+3*cameraID;
      double* cameraMat_3d = location_3d+3*cameraID;
      if (!(std::isnan(*cameraPoint))){
          cameraPoint[0] = cameraMat_3d[0];
          cameraPoint[1] = cameraMat_3d[1];
          cameraPoint[2] = cameraMat_3d[2];
          //std::cout<<"cameraID="<<cameraID<<" : ";
          //std::cout<<"cameraPoint="<<cameraPoint[0]<<" "<<cameraPoint[1]<<std::endl;
      }
  }	 

	auto buf5 = x5.request();
	double* correspondence_new = static_cast<double*>(buf5.ptr);	
    double* correspondence = new double [nObs];
    for(int cameraID=0; cameraID<nObs; ++cameraID){
		double* correspondencePtr = correspondence+cameraID;
		double* correspondenceMat = correspondence_new+cameraID;
		correspondencePtr[0] = correspondenceMat[0];
	}

	  // construct camera parameters from camera matrix
      auto buf7 = x7.request();
	  double* carRt = static_cast<double*>(buf7.ptr);
	  double* carPtr = new double [6*nCam];
	  for(int cameraID=0; cameraID<nCam; ++cameraID)
	  {		
		  double* carPtr1 = carPtr+6*cameraID;
		  double* carMat = carRt+12*cameraID;
		  ceres::RotationMatrixToAngleAxis<double>(carMat, carPtr1);
		  carPtr1[3] = carMat[9];
		  carPtr1[4] = carMat[10];
		  carPtr1[5] = carMat[11];
		  //std::cout<<"cameraID="<<cameraID<<" : ";
		  //std::cout<<"car="<<carPtr1[0]<<" "<<carPtr1[1]<<" "<<carPtr1[2]<<" "<<carPtr1[3]<<" "<<carPtr1[4]<<" "<<carPtr1[5]<<std::endl;

	  }            

	auto buf8 = x8.request();
	double* time_instance_new = static_cast<double*>(buf8.ptr);	
    double* time_instance = new double [nObs];
    for(int cameraID=0; cameraID<nObs; ++cameraID){
		double* time_instancePtr = time_instance+cameraID;
		double* time_instanceMat = time_instance_new+cameraID;
        if (!(std::isnan(*time_instancePtr))){
		time_instancePtr[0] = time_instanceMat[0];
		}

	}

    double* alpha = read_data(x6);



	  ceres::Problem problem;
	  ceres::LossFunction* loss_function = NULL; // squared loss    
	  double* carPtr_check = new double [6];
	  carPtr_check[0] = 0;carPtr_check[1] = 0;carPtr_check[2] = 0;carPtr_check[3] = 0;carPtr_check[4] = 0;carPtr_check[5] = 0;
//	  for (unsigned int idObs=0; idObs<nObs; ++idObs){
	  for (unsigned int idObs=0; idObs<nObs; ++idObs){
		double* cameraPtr_ceres = cameraParameter + idObs * 12;
		double* cameraInt_ceres = cameraIntrinscis + idObs * 9;
		double* observePtr = camera_2d+2*idObs;
		int counter = (int) correspondence[idObs];
		double* pointPtr  = pointCloud + 3*counter;
		
		int time = (int) time_instance[idObs];
		double* carPtr_time = carPtr + time * 6;
		//std::cout<<"cameraPtr="<<" "<<observePtr[1];
		//std::cout<<" "<<cameraPtr_ceres[0];
		//std::cout<<" "<<cameraInt_ceres[0];
		//std::cout<<" "<<pointPtr[0];
		//std::cout<<" "<<carPtr_time[0];
		//std::cout<<" "<<alpha[0];
		//std::cout<<" "<< carPtr[4]<<" "<<sizeof(carPtr_time)<<std::endl;
		
		ceres::CostFunction* cost_function;
        cost_function = new ceres::AutoDiffCostFunction<AlignmentErrorCar, 2, 6,1>(new AlignmentErrorCar(observePtr,cameraPtr_ceres,cameraInt_ceres,pointPtr));
        problem.AddResidualBlock(cost_function,loss_function,carPtr_time,alpha);
        

		}
		for (int time_loop =1;time_loop <  (int) *std::max_element(time_instance,time_instance+nObs) - 1;time_loop++)
		{	
			//std::cout << *std::max_element(time_instance,time_instance+nObs)<<" "<<nObs<<"\n";
			double* carPtr_time = carPtr + (time_loop) * 6;
			double* carPtr_time_old = carPtr + (time_loop-1) * 6;
			double* carPtr_time_new = carPtr + (time_loop+1) * 6;
			ceres::CostFunction* cost_function_traj;
			ceres::LossFunction* loss_function_traj = NULL; // squared loss    
			cost_function_traj = new ceres::AutoDiffCostFunction<AlignmentErrorTrajectory,3,6,6,6>(new AlignmentErrorTrajectory());
			problem.AddResidualBlock(cost_function_traj,loss_function_traj,carPtr_time_old,carPtr_time,carPtr_time_new);
		}
		 ceres::Solver::Options options;
		 options.max_num_iterations = 1000;  
		 options.minimizer_progress_to_stdout = true;
		 options.linear_solver_type = ceres::SPARSE_SCHUR;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;
		  ////options.ordering_type = ceres::SCHUR;
		 ceres::Solver::Summary summary;
		 ceres::Solve(options, &problem, &summary);
	    std::cout << summary.BriefReport() << "\n";
      //std::cout<<alpha[0]<<" "<<carPtr_check[1]<<" "<<carPtr_check[2]<<" "<<carPtr_check[3]<<" "<<carPtr_check[4]<<" "<<carPtr_check[5]<<std::endl;
      std::stringstream ss;
	  
	  for(int cameraID=0; cameraID<nCam; ++cameraID)
	  {		
		  double* carPtr_new = carPtr+6*cameraID;
		  double* carMat_op_new = carRt+12*cameraID;
		  ceres::AngleAxisToRotationMatrix<double>(carPtr_new,carMat_op_new);
	      carMat_op_new[9] = carPtr_new[3];
	      carMat_op_new[10] = carPtr_new[4];
	      carMat_op_new[11] = carPtr_new[5];
	      
	      
    for(size_t i = 0; i < 12; i++)
    {
		//std::cout<<i<<std::endl;
			if(i != 0)
				ss << ",";
			ss << carMat_op_new[i];
		}
  	if(cameraID <nCam-1)
		ss << ",";
	  
	  }
     std::string s = ss.str();
     //std::cout<<s<<std::endl;
     
      //std::cout<<"carRt_op="<<anCamlpha[0]<<" "<<carRt_op[0]<<" "<<carRt_op[1]<<" "<<carRt_op[2]<<" "<<carRt_op[3]<<" "<<carRt_op[4]<<" "<<carRt_op[9]<<std::endl;






    ////ceres::Solve(options, problem, data, &summary);


    
    //auto iterations = summary.iterations;
    //int nfev = 0;
    //int ngev = 0;
    //for (auto& summ: iterations) {
        //nfev += summ.line_search_function_evaluations;
        //ngev += summ.line_search_gradient_evaluations;
    //}

    auto OptimizeResult = py::module::import("scipy.optimize").attr("OptimizeResult");

    py::dict out("scale"_a = alpha[0],
				 "rt"_a = s,
                 "success"_a = summary.termination_type == ceres::CONVERGENCE ||
                               summary.termination_type == ceres::USER_SUCCESS,
                 "status"_a = (int)summary.termination_type,
                 "message"_a = summary.message,
                 "fun"_a = summary.final_cost
                 );

	//std::cout<<"check"<<std::endl;
    //py::dict out("x"_a = pointCloud[0],
				 //"y"_a = pointCloud[1],
				 //"z"_a = pointCloud[2],
				 //"rt00"_a = carRt_op[0],
				 //"rt01"_a = carRt_op[1],
				 //"rt02"_a = carRt_op[2],
				 //"rt10"_a = carRt_op[3],
				 //"rt11"_a = carRt_op[4],
				 //"rt12"_a = carRt_op[5],
				 //"rt20"_a = carRt_op[6],
				 //"rt21"_a = carRt_op[7],
				 //"rt22"_a = carRt_op[8],
				 //"rt03"_a = carPtr[3],
				 //"rt13"_a = carPtr[4],
				 //"rt23"_a = carPtr[5],
				 //"scale"_a = alpha[0],
				 //"rt"_a = s,
                 //"success"_a = summary.termination_type == ceres::CONVERGENCE ||
                               //summary.termination_type == ceres::USER_SUCCESS,
                 //"status"_a = (int)summary.termination_type,
                 //"message"_a = summary.message,
                 //"fun"_a = summary.final_cost
                 //);
    std::cout<<"check"<<std::endl;

    return OptimizeResult(out);
    //return x0;
}


namespace py = pybind11;

PYBIND11_PLUGIN(ceres) {
    py::module m("ceres", "Python bindings to the Ceres-Solver minimizer.");
    google::InitGoogleLogging("ceres");

    //m.def("optimize", &optimize, "Optimizes the function");
    m.def("ba_optimize", &ba_optimize, "Optimizes the function");
    m.def("ba_optimize_video", &ba_optimize_video, "Optimizes the function");

    return m.ptr();
}
