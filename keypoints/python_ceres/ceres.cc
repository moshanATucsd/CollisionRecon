#include <iostream>
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <glog/logging.h>
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

struct AlignmentErrorCar {
  AlignmentErrorCar(double* observed_in, double* camera_extrinsic_in, double* camera_intrinsic_in, double* keyp_3d_in): observed(observed_in), camera_extrinsic(camera_extrinsic_in),camera_intrinsic(camera_intrinsic_in),keyp_3d(keyp_3d_in) {}

  template <typename T>
  bool operator()(const T* const car_extrinsic,const T* const alpha, T* residuals) const {
                  
    // camera_extrinsic[0,1,2] are the angle-axis rotation.
    T p[3];
    T o[3];
    o[0] = alpha[0]*keyp_3d[0];
    o[1] = alpha[0]*keyp_3d[1];
    o[2] = alpha[0]*keyp_3d[2];
    ceres::AngleAxisRotatePoint(car_extrinsic, o, p);
    
    p[0] = T(camera_extrinsic[0])*p[0] + T(camera_extrinsic[1])*p[1] + T(camera_extrinsic[2])*p[2];
    p[1] = T(camera_extrinsic[3])*p[0] + T(camera_extrinsic[4])*p[1] + T(camera_extrinsic[5])*p[2];
    p[2] = T(camera_extrinsic[6])*p[0] + T(camera_extrinsic[7])*p[1] + T(camera_extrinsic[8])*p[2];
  
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

    
    std::cout<<"p[0]="<<p[0]<<std::endl;
    std::cout<<"p[1]="<<p[1]<<std::endl;
    std::cout<<"observed[0]="<<observed[0]<<std::endl;
    std::cout<<"observed[1]="<<observed[1]<<std::endl;
    std::cout<<"residuals[0]="<<residuals[0]<<std::endl;
    std::cout<<"residuals[1]="<<residuals[1]<<std::endl;
    std::cout<<"camera_intrinsic[0]="<<car_extrinsic[3]<<std::endl;
    std::cout<<"camera_extrinsic[3]="<<car_extrinsic[4]<<std::endl;
    std::cout<<"--------------------------"<<std::endl;
    
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


py::object ba_optimize(py::function func, py::function grad, py::buffer x0,py::buffer x1,py::buffer x2,py::buffer x3,py::buffer x4,py::buffer x5,py::buffer x6,py::buffer x7) {
	double* data = read_data(x0);
	int mode = data[0];
	int nCam = data[1];
	int nPts = data[2];
	int nObs = data[3];



  ////// construct camera parameters from camera matrix
  double* cameraRt = read_data(x1);
  double* cameraParameter = new double [12*nCam];
  for(int cameraID=0; cameraID<nCam; ++cameraID){
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
          std::cout<<"cameraID="<<cameraID<<" : ";
          std::cout<<"cameraPtr="<<cameraPtr[0]<<" "<<cameraPtr[1]<<" "<<cameraPtr[2]<<" "<<cameraPtr[3]<<" "<<cameraPtr[4]<<" "<<cameraPtr[5]<<std::endl;
      }
  }	  
  
  double* cameraK = read_data(x2);
  double* cameraIntrinscis = new double [9*nCam];
  for(int cameraID=0; cameraID<nCam; ++cameraID){
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
          std::cout<<"cameraID="<<cameraID<<" : ";
          std::cout<<"cameraInt="<<cameraInt[0]<<" "<<cameraInt[1]<<" "<<cameraInt[2]<<" "<<cameraInt[3]<<" "<<cameraInt[4]<<" "<<cameraInt[5]<<std::endl;
      }
  }	  

  double* location_2d = read_data(x3);
  double* camera_2d = new double [2*nCam];
  for(int cameraID=0; cameraID<nCam; ++cameraID){
      double* cameraPtr_2d = camera_2d+2*cameraID;
      double* cameraMat_2d = location_2d+2*cameraID;
      if (!(std::isnan(*cameraPtr_2d))){
          cameraPtr_2d[0] = cameraMat_2d[0];
          cameraPtr_2d[1] = cameraMat_2d[1];
          std::cout<<"cameraID="<<cameraID<<" : ";
          std::cout<<"cameraPtr_2d="<<cameraPtr_2d[0]<<" "<<cameraPtr_2d[1]<<std::endl;
      }
  }	
  
  double* location_3d = read_data(x4);	 
  double* pointCloud = new double [3*nPts];
  for(int cameraID=0; cameraID<nPts; ++cameraID){
      double* cameraPoint = pointCloud+3*cameraID;
      double* cameraMat_3d = location_3d+3*cameraID;
      if (!(std::isnan(*cameraPoint))){
          cameraPoint[0] = cameraMat_3d[0];
          cameraPoint[1] = cameraMat_3d[1];
          cameraPoint[2] = cameraMat_3d[2];
          std::cout<<"cameraID="<<cameraID<<" : ";
          std::cout<<"cameraPoint="<<cameraPoint[0]<<" "<<cameraPoint[1]<<std::endl;
      }
  }	 

	double* correspondence = read_data(x5);

    double* alpha = read_data(x6);

	  // construct camera parameters from camera matrix
	  double* carRt = read_data(x7);
	  double* carPtr = new double [6];
      std::cout<<"carPtr="<<alpha[0]<<" "<<carRt[0]<<" "<<carRt[1]<<" "<<carRt[2]<<" "<<carRt[3]<<" "<<carRt[4]<<" "<<carRt[5]<<std::endl;
      ceres::RotationMatrixToAngleAxis<double>(carRt, carPtr);
      carPtr[3] = carRt[9];
      carPtr[4] = carRt[10];
      carPtr[5] = carRt[11];
      

	  // output info
	  std::cout<<"Parameters: ";
	  std::cout<<"mode="<<mode<<" ";

	  std::cout<<"Meta Info: ";
	  std::cout<<"nCam="<<nCam<<" ";
	  std::cout<<"nPts="<<nPts<<" ";
	  std::cout<<"nObs="<<nObs<<"\t"; //<<std::endl;

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
		double* cameraPtr = cameraParameter + idObs * 12;
		double* cameraInt = cameraIntrinscis + idObs * 9;
		double* observePtr = camera_2d+2*idObs;
		int counter = (int) correspondence[idObs];
		double* pointPtr  = pointCloud + 3*counter;
        std::cout<<"cameraPtr="<<pointPtr[1000]<<" "<<pointPtr[1]<<" "<<pointPtr[2]<<std::endl;
		  
		ceres::CostFunction* cost_function;
        //cost_function = new ceres::AutoDiffCostFunction<AlignmentErrorTriangulate, 2, 3>(new AlignmentErrorTriangulate(observePtr,cameraPtr,cameraInt));
        //problem.AddResidualBlock(cost_function,loss_function,pointPtr);
        cost_function = new ceres::AutoDiffCostFunction<AlignmentErrorCar, 2, 6,1>(new AlignmentErrorCar(observePtr,cameraPtr,cameraInt,pointPtr));
        problem.AddResidualBlock(cost_function,loss_function,carPtr,alpha);
		}
		
		 ceres::Solver::Options options;
		 options.max_num_iterations = 200;  
		 options.minimizer_progress_to_stdout = true;
		 options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;
		  ////options.ordering_type = ceres::SCHUR;
		 ceres::Solver::Summary summary;
		 ceres::Solve(options, &problem, &summary);
	  double* carRt_op = new double [9];
      ceres::AngleAxisToRotationMatrix<double>(carPtr,carRt_op);

      std::cout<<"carRt_op="<<alpha[0]<<" "<<carRt_op[0]<<" "<<carRt_op[1]<<" "<<carRt_op[2]<<" "<<carRt_op[3]<<" "<<carRt_op[4]<<" "<<carRt_op[5]<<std::endl;






    ////ceres::Solve(options, problem, data, &summary);


    
    //auto iterations = summary.iterations;
    //int nfev = 0;
    //int ngev = 0;
    //for (auto& summ: iterations) {
        //nfev += summ.line_search_function_evaluations;
        //ngev += summ.line_search_gradient_evaluations;
    //}

    auto OptimizeResult = py::module::import("scipy.optimize").attr("OptimizeResult");

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
                 "success"_a = summary.termination_type == ceres::CONVERGENCE ||
                               summary.termination_type == ceres::USER_SUCCESS,
                 "status"_a = (int)summary.termination_type,
                 "message"_a = summary.message,
                 "fun"_a = summary.final_cost
                 );

    return OptimizeResult(out);
    //return x0;
}

namespace py = pybind11;

PYBIND11_PLUGIN(ceres) {
    py::module m("ceres", "Python bindings to the Ceres-Solver minimizer.");
    google::InitGoogleLogging("ceres");

    //m.def("optimize", &optimize, "Optimizes the function");
    m.def("ba_optimize", &ba_optimize, "Optimizes the function");

    return m.ptr();
}
