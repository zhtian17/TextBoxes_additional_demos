#include <Python.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//struct sample
//{
//	Mat img;
//	string img_name;
//};

int load_network(const char* name, vector<PyObject*> &result)
{
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");

	PyObject *pName, *pModule, *pFunc, *pArgs;
	
	pName = PyString_FromString(name);
	pModule = PyImport_Import(pName);
	
	if (!pModule)
	{
		cout << "error: can't find python module" << endl;
		return -1;
	}

	PyObject* pDict = PyModule_GetDict(pModule);
	if (!pDict){
		return -1;
	}

	pFunc = PyDict_GetItemString(pDict, "get_config");
	if (!pFunc || !PyCallable_Check(pFunc)){
		cout << "error: can't find function [get_config]" << endl;
		getchar();
		return -1;
	}

	pArgs = NULL;
	PyObject* config = PyObject_CallObject(pFunc, pArgs);

	pFunc = PyDict_GetItemString(pDict, "prepare_network");
	if (!pFunc || !PyCallable_Check(pFunc)) {
		cout << "error: can't find function [prepare_network]" << endl;
		getchar();
		return -1;
	}

	PyObject* net = PyObject_CallFunctionObjArgs(pFunc, config,NULL);
	if (!net)
	{
		cout << "error: get net failed" << endl;
		return -1;
	}
	
	result.push_back(pDict);
	result.push_back(config);
	result.push_back(net);

	Py_XDECREF(pName);
	Py_XDECREF(pArgs);
	Py_XDECREF(pFunc);
	Py_XDECREF(pModule);

	return 0;
}

int text_detection(vector<Mat> vec_spl, PyObject *pDict, PyObject *config, PyObject *net)
{
	PyObject *pFunc;
	pFunc = PyDict_GetItemString(pDict, "detection");
	if (!pFunc || !PyCallable_Check(pFunc)) {
		cout << "can't find function [detection]" << endl;
		getchar();
		return -1;
	}

	for (int i = 0; i < vec_spl.size(); i++)
	{
		Mat img;
		//vec_spl[i].img.copyTo(img);
		vec_spl[i].copyTo(img);
		int width = img.cols, height = img.rows;
		PyObject* img_bytearray = NULL;
		img_bytearray = PyByteArray_FromStringAndSize((char*)img.data, width*height*3);
		if (!img_bytearray) {
			cout << "can't pass image to detection" << endl;
			return -1;
		}
		
		//PyObject *img_name_py = Py_BuildValue("s", vec_spl[i].img_name.data());
		PyObject *img_num_py = Py_BuildValue("i", i);
		PyObject *width_py = Py_BuildValue("i", width);
		PyObject *height_py = Py_BuildValue("i", height);
		PyObject *result = PyObject_CallFunctionObjArgs(pFunc, config, net,img_bytearray,width_py, height_py, img_num_py, NULL);
		if (result)

		Py_XDECREF(img_bytearray);
		Py_XDECREF(img_num_py);
		Py_XDECREF(width_py);
		Py_XDECREF(height_py);
		Py_XDECREF(result);
	}
	Py_XDECREF(pFunc);
	return 0;
}

int text_detection_0(PyObject *pDict, PyObject *config, PyObject *net)
{
	PyObject *pFunc;
	pFunc = PyDict_GetItemString(pDict, "detection");
	if (!pFunc || !PyCallable_Check(pFunc)) {
		cout << "can't find function [detection]" << endl;
		getchar();
		return -1;
	}

	PyObject *result = PyObject_CallFunctionObjArgs(pFunc, config, net, NULL);
	if (result)

	Py_XDECREF(result);
	Py_XDECREF(pFunc);
	return 0;
}

int prepare_image(Mat std, Mat gerber_silk, vector<Mat> &vec_spl)
{
	const int num_patch_x = 4;
	const int num_patch_y = 4;

	int std_width = std.cols;
	int std_height = std.rows;

	Mat gerber_temp;
	gerber_silk.convertTo(gerber_temp, CV_32FC1);

	//find the border according to the coordinated of pads
	Mat col_sum(1, std_width, CV_32FC1);
	Mat row_sum(std_height, 1, CV_32FC1);

	reduce(gerber_temp, col_sum, 0, CV_REDUCE_SUM);
	reduce(gerber_temp, row_sum, 1, CV_REDUCE_SUM);

	int min_x = 0, max_x = std_width - 1;
	while (col_sum.at<float>(0, min_x) == 0)
		min_x++;
	while (col_sum.at<float>(0, max_x) == 0)
		max_x--;

	int min_y = 0, max_y = std_height - 1;
	while (row_sum.at<float>(min_y, 0) == 0)
		min_y++;
	while (row_sum.at<float>(max_y, 0) == 0)
		max_y--;

	int roi_width = max_x - min_x + 1;
	int roi_height = max_y - min_y + 1;

	const int border_x = roi_width / num_patch_x / 5;
	const int border_y = roi_height / num_patch_y / 5;

	min_x = max(min_x - border_x, 0);
	max_x = min(max_x + border_x, std_width);
	min_y = max(min_y - border_y, 0);
	max_y = min(max_y + border_y, std_height);

	roi_width = max_x - min_x + 1;
	roi_height = max_y - min_y + 1;

	int step_x = roi_width / num_patch_x;
	int step_y = roi_height / num_patch_y;

	for (int idx_y = 0; idx_y < num_patch_y; idx_y++)
	{
		int y_st = min_y + idx_y * step_y;
		int y_ed = min(min_y + (idx_y + 1)*step_y, std_height);

		for (int idx_x = 0; idx_x < num_patch_x; idx_x++)
		{
			int x_st = min_x + idx_x * step_x;
			int x_ed = min(min_x + (idx_x + 1)*step_x, std_width);

			Mat patch;
			std(Range(y_st, y_ed), Range(x_st, x_ed)).copyTo(patch);

			vec_spl.push_back(patch);
		}
	}
	return 0;
}

int main() 
{
	string dir = "/home/zhout/TextBoxes_plusplus/demo_images/standard_images/";
	//ifstream list;
	//list.open("/home/zhout/TextBoxes_plusplus/demo_images/test/list.txt", ios::in);
	string img_name = "KIBANCUR.bmp";
	string gerber_name = "gerber_silk.bmp";
	Mat std = imread(dir + img_name, IMREAD_UNCHANGED);
	Mat gerber_silk = imread(dir + gerber_name, IMREAD_UNCHANGED);
	if (std.empty() || gerber_silk.empty())
	{
		cout << "cannot open image" << endl;
		getchar();
	}
	//vector<sample> vec_spl;
	vector<Mat> vec_spl;
	prepare_image(std, gerber_silk, vec_spl);

	//while (!list.eof())
	//{
	//	sample spl;
	//	string name;
	//	getline(list, name);
	//	if (name == "")
	//		break;
	//	Mat img = imread(dir + name, IMREAD_UNCHANGED);

	//	spl.img_name = name;
	//	spl.img = img;
	//	vec_spl.push_back(spl);
	//}
	//list.close();

	Py_Initialize();
	if (!Py_IsInitialized())
		return -1;

	const char *net_name = "textpy";

	vector<PyObject*> res_net;
	load_network(net_name, res_net);

	PyObject *pDict = res_net[0];
	PyObject *config = res_net[1];
	PyObject *net = res_net[2];
	if (!net)
	{
		cout << "error: can't get net" << endl;
		return -1;
	}

	text_detection(vec_spl, pDict, config, net);
	//text_detection_0(pDict, config, net);

	Py_XDECREF(pDict);
	Py_XDECREF(config);
	Py_XDECREF(net);

	Py_Finalize();

	cout << "Done!" << endl;
	return 0;
}
