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

PyObject* text_detection(vector<sample> vec_spl, PyObject *pDict, PyObject *config, PyObject *net)
{
	PyObject *result = NULL;
	int img_num = vec_spl.size();
		
	PyObject *pFunc;
	pFunc = PyDict_GetItemString(pDict, "detection");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		cout << "can't find function [detection]" << endl;
		getchar();
		return NULL;
	}

	int width = vec_spl[0].img.cols;
	int height = vec_spl[0].img.rows;

	PyObject *imgs_py = PyList_New(vec_spl.size());
	PyObject *imgs_name_py = PyList_New(vec_spl.size());
	for (int i = 0; i < img_num; i++)
	{
		//PyObject* img_bytearray = PyByteArray_FromStringAndSize((char*)vec_spl[i].img.data, width*height * 3);
		/*if (!img_bytearray) 
		{
			cout << "can't pass image to detection" << endl;
			return NULL;
		}*/
		PyList_SetItem(imgs_py, i, PyByteArray_FromStringAndSize((char*)vec_spl[i].img.data, width*height * 3));
		PyList_SetItem(imgs_name_py, i, Py_BuildValue("s", vec_spl[i].img_name.data()));

		//Py_XDECREF(img_bytearray);
	}
	//PyObject *img_num_py = Py_BuildValue("i", i);
	PyObject *width_py = Py_BuildValue("i", width);
	PyObject *height_py = Py_BuildValue("i", height);
	PyObject *img_num_py = Py_BuildValue("i", img_num);
	result = PyObject_CallFunctionObjArgs(pFunc, config, net, imgs_py, width_py, height_py, imgs_name_py, img_num_py, NULL);

	Py_XDECREF(imgs_name_py);
	Py_XDECREF(imgs_py);
	Py_XDECREF(width_py);
	Py_XDECREF(height_py);
	Py_XDECREF(pFunc);
	return result;
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
	PyObject *result = NULL;
	if (vec_spl.size() != 0)
	{
		result = text_detection(vec_spl, pDict, config, net);
	}
	else
	{
		cout<<"No samples"<<endl;
	}

	Py_XDECREF(pDict);
	Py_XDECREF(config);
	Py_XDECREF(net);
	Py_XDECREF(result);

	Py_Finalize();

	cout << "Done!" << endl;
	return 0;
}
