#include <Python.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct sample
{
	Mat img;
	string img_name;
};

int load_network(const char* name, PyObject* config_info, vector<PyObject*> &result)
{
	//This function sets configuration and prepares the whole network.
	//Please initialize python using Py_Initialize before calling this function.
	//input:	-name : 				the name of python module
	//				-config_info:		the configureation information, which is a python tuple
	//output:	-result:				a vector of python object, [pDict, config, net]

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");

	PyObject *pName, *pModule, *pFunc;

	pName = PyString_FromString(name);
	pModule = PyImport_Import(pName);

	if (!pModule)
	{
		cout << "error: can't find python module" << endl;
		return -1;
	}

	//Get dictionary of python functions
	PyObject* pDict = PyModule_GetDict(pModule);
	if (!pDict){
		return -1;
	}

	//Get get_config function
	pFunc = PyDict_GetItemString(pDict, "get_config");
	if (!pFunc || !PyCallable_Check(pFunc)){
		cout << "error: can't find function [get_config]" << endl;
		getchar();
		return -1;
	}
	//Call get_config with arguments
	PyObject* config = PyObject_CallFunctionObjArgs(pFunc, config_info, NULL);

	//Get prepare_network function
	pFunc = PyDict_GetItemString(pDict, "prepare_network");
	if (!pFunc || !PyCallable_Check(pFunc)) {
		cout << "error: can't find function [prepare_network]" << endl;
		getchar();
		return -1;
	}
	//Call prepare_network function with arguments
	PyObject* net = PyObject_CallFunctionObjArgs(pFunc, config, NULL);
	if (!net)
	{
		cout << "error: get net failed" << endl;
		return -1;
	}

	result.push_back(pDict);
	result.push_back(config);
	result.push_back(net);

	//Delete pointers
	Py_XDECREF(pName);
	Py_XDECREF(pFunc);
	Py_XDECREF(pModule);

	return 0;
}

int text_detection(vector<sample> vec_spl, PyObject *pDict, PyObject *config, PyObject *net, vector<vector<double> > &result)
{
	// This function performs text detetion on a vector of images.
	//input:	-vec_spl: a vector samples, the structure sample contains both the image and its name
	//				-pDict:		the dictionary of python functions
	//				-config:  the configuration getten from prepare_network
	//				-net:			the prepared net getten from prepare_network
	//output:	-result:	the 2-dim vector of detecting result, in this case, [img_idx,x1,y1,x2,y2,x3,y3,x4,y4,score]

	int img_num = vec_spl.size();

	//Get detection function
	PyObject *pFunc;
	pFunc = PyDict_GetItemString(pDict, "detection");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		cout << "can't find function [detection]" << endl;
		getchar();
		return -1;
	}

	int width = vec_spl[0].img.cols;
	int height = vec_spl[0].img.rows;

	//Convert vector into python list
	PyObject *imgs_py = PyList_New(vec_spl.size());
	PyObject *imgs_name_py = PyList_New(vec_spl.size());
	for (int i = 0; i < img_num; i++)
	{
		PyList_SetItem(imgs_py, i, PyByteArray_FromStringAndSize((char*)vec_spl[i].img.data, width*height * 3));
		PyList_SetItem(imgs_name_py, i, Py_BuildValue("s", vec_spl[i].img_name.data()));
	}
	//Convert int into python integer
	PyObject *width_py = Py_BuildValue("i", width);
	PyObject *height_py = Py_BuildValue("i", height);
	PyObject *img_num_py = Py_BuildValue("i", img_num);
	//Call detection
	PyObject *result_py = PyObject_CallFunctionObjArgs(pFunc, config, net, imgs_py, width_py, height_py, imgs_name_py, img_num_py, NULL);
	int result_m = PyList_Size(result_py);
	int result_n = PyList_Size(PyList_GetItem(result_py,0));
	for (int i = 0; i < result_m; i++)
	{
		vector<double> res_temp;
		for (int j = 0; j < result_n; j++)
		{
			double temp = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(result_py,i),j));
			res_temp.push_back(temp);
		}
		result.push_back(res_temp);
	}

	//Delete pointers
	Py_XDECREF(pFunc);
	Py_XDECREF(imgs_py);
	Py_XDECREF(imgs_name_py);
	Py_XDECREF(width_py);
	Py_XDECREF(height_py);
	Py_XDECREF(img_num_py);
	Py_XDECREF(result_py);
	return 0;
}

int main()
{
	string dir = "/home/zhout/TextBoxes_plusplus/demo_images/test/";
	ifstream list;
	list.open("/home/zhout/TextBoxes_plusplus/demo_images/test/list.txt", ios::in);

	//Load images and form a vector
	vector<sample> vec_spl;
	while (!list.eof())
	{
		sample spl;
		string name;
		getline(list, name);
		if (name == "")
			break;
		Mat img = imread(dir + name);
		if (img.empty())
		{
			cout << "Cannot read image" << endl;
			return -1;
		}
		spl.img_name = name;
		spl.img = img;
		vec_spl.push_back(spl);
	}
	list.close();

	//Initialize python
	Py_Initialize();
	if (!Py_IsInitialized())
		return -1;

	const char *net_name = "textpy";
	PyObject* config_info = PyTuple_New(12);
	// this file is expected to be in {caffe_root}/examples
	PyTuple_SetItem(config_info, 0, Py_BuildValue("s", "../../"));
	PyTuple_SetItem(config_info, 1, Py_BuildValue("s", "./models/deploy.prototxt")); //model_def
	PyTuple_SetItem(config_info, 2, Py_BuildValue("s", "./models/model_icdar15.caffemodel")); //model_weights
	PyTuple_SetItem(config_info, 3, Py_BuildValue("s", "./demo_images/test_result_img/")); //det_visu_dir
	PyTuple_SetItem(config_info, 4, Py_BuildValue("s", "./demo_images/test_result_txt/")); //det_save_dir
	PyTuple_SetItem(config_info, 5, Py_BuildValue("s", "./demo_images/crops/")); //crop_dir
	PyTuple_SetItem(config_info, 6, Py_BuildValue("s", "./crnn/data/icdar_generic_lexicon.txt")); //lexicon_path
	PyTuple_SetItem(config_info, 7, Py_BuildValue("i", 1)); //use_lexcion=True
	PyTuple_SetItem(config_info, 8, Py_BuildValue("f", 0.2)); //overlap_threshold
	PyTuple_SetItem(config_info, 9, Py_BuildValue("f", 0.5)); //det_score_threshold
	PyTuple_SetItem(config_info, 10, Py_BuildValue("f", 0.7)); //f_score_threshold
	PyTuple_SetItem(config_info, 11, Py_BuildValue("i", 1)); //visu_detection = True

	//Load the network
	vector<PyObject*> res_net;
	load_network(net_name, config_info, res_net);
	PyObject *pDict = res_net[0];
	PyObject *config = res_net[1];
	PyObject *net = res_net[2];
	if (!net)
	{
		cout << "error: can't get net" << endl;
		return -1;
	}

	//Perform detection
	vector<vector<double> > result;
	if (vec_spl.size() != 0)
	{
		text_detection(vec_spl, pDict, config, net, result);
		cout << "finish detection" << endl;
	}
	else
		cout<<"No samples"<<endl;

	Py_XDECREF(pDict);
	Py_XDECREF(config);
	Py_XDECREF(net);

	//Finalize python
	Py_Finalize();

	cout << "Done!" << endl;
	return 0;
}
