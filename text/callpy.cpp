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

int text_detection(vector<sample> vec_spl, PyObject *pDict, PyObject *config, PyObject *net)
{
	int width = 500, height = 500;

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
		vec_spl[i].img.copyTo(img);
		PyObject* img_bytearray = NULL;
		img_bytearray = PyByteArray_FromStringAndSize((char*)img.data, width*height*3);
		if (!img_bytearray) {
			cout << "can't pass image to detection" << endl;
			return -1;
		}
		
		PyObject *img_name_py = Py_BuildValue("s", vec_spl[i].img_name.data());
		PyObject *result = PyObject_CallFunctionObjArgs(pFunc, config, net,img_bytearray,img_name_py, NULL);
		if (result)

		Py_XDECREF(img_bytearray);
		Py_XDECREF(img_name_py);
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

int main() 
{
	string dir = "/home/zhout/TextBoxes_plusplus/demo_images/test/";
	ifstream list;
	list.open("/home/zhout/TextBoxes_plusplus/demo_images/test/list.txt", ios::in);

	vector<sample> vec_spl;
	while (!list.eof())
	{
		sample spl;
		string name;
		getline(list, name);
		if (name == "")
			break;
		Mat img = imread(dir + name, IMREAD_UNCHANGED);

		spl.img_name = name;
		spl.img = img;
		vec_spl.push_back(spl);
	}
	list.close();

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
