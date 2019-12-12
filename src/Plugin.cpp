
#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // XCode does not need annotating exported functions, so define is empty
#endif

// ------------------------------------------------------------------------
// Plugin itself

#include <vector>
#include <memory>
#include "regression.h"
#include "classification.h"
#include "seriesClassification.h"


std::vector < std::unique_ptr <regression> > regression_models;
std::vector < std::unique_ptr <std::vector<trainingExample>> > training_sets;

// Link following functions C-style (required for plugins)
extern "C"
{

#pragma region Test Method

	//Exported method that adds a parameter text to an additional text and returns them combined.
	EXPORT_API char *getCharArray(char* parameterText)
	{
		const char* additionalText = " world!";

		if (strlen(parameterText) + strlen(additionalText) + 1 > 256)
		{
			const char* message = "Error: Maximum size of the char array is 256 chars.";
			return (char*)message;
		}

		char combinedText[256] = "";

		strcpy_s(combinedText, 256, parameterText);
		strcat_s(combinedText, 256, additionalText);

		return (char*)combinedText;
	}

#pragma region Model Creation

	EXPORT_API regression * createRegressionModel() {
		//regression_models.push_back(std::unique_ptr<regression>(new regression(3, 4)));
		//return regression_models.back().get();
		return new regression();
	}

	EXPORT_API classification * createClassificationModel() {
		//regression_models.push_back(std::unique_ptr<regression>(new regression(3, 4)));
		//return regression_models.back().get();
		return new classification();
	}

	EXPORT_API seriesClassification * createSeriesClassificationModel() {
		return new seriesClassification();
	}

#pragma endregion


#pragma region Model Destruction

	EXPORT_API void destroyModel(modelSet<double> *model) {
		delete model;
	}

	EXPORT_API void destroySeriesClassificationModel(seriesClassification *model) {
		delete model;
	}

#pragma endregion



#pragma region Model JSON Configuration

	EXPORT_API const char * getJSON(modelSet<double> *model) {
		std::string jsonString = model->getJSON();
		char * jsonCString = new char[jsonString.size() + 1];
		std::copy(jsonString.begin(), jsonString.end(), jsonCString);
		jsonCString[jsonString.size()] = '\0';
		return jsonCString;
		//return ::SysAllocString(L"Greetings from the native world!");
		//return "hello";
	}

	EXPORT_API void putJSON(modelSet<double> *model, const char *jsonString) {
		model->putJSON(jsonString);
	}

#pragma endregion


#pragma region Training Data 

	/* TRAINING EXAMPLES */

	EXPORT_API std::vector<trainingExample > * createTrainingSet() {
		//training_sets.push_back(std::unique_ptr<std::vector<trainingExample >>(new std::vector<trainingExample >()));
		//return training_sets.back().get();
		return new std::vector<trainingExample>();
	}

	EXPORT_API void destroyTrainingSet(std::vector<trainingExample> *trainingSet) {
		delete trainingSet;
	}

	EXPORT_API void addTrainingExample(std::vector<trainingExample> *trainingSet, double *inputs, int numInputs, double *outputs, int numOutputs) {
		trainingExample tempExample;
		//tempExample.input = { 0.2, 0.7 };
		for (int i = 0; i < numInputs; i++) {
			tempExample.input.push_back(inputs[i]);
		}
		for (int i = 0; i < numOutputs; i++) {
			tempExample.output.push_back(outputs[i]);
		}
		//tempExample.input.insert(tempExample.input.begin(), inputs, inputs+numInputs);
		//tempExample.output = { 3.0 };
		//tempExample.output.insert(tempExample.input.begin(), outputs, outputs + numOutputs);
		trainingSet->push_back(tempExample);
	}

	EXPORT_API int getNumTrainingExamples(std::vector<trainingExample> *trainingSet) {
		return trainingSet->size();
	}

	EXPORT_API double getInput(std::vector<trainingExample> *trainingSet, int i, int j) {
		if (i < trainingSet->size() && j < (*trainingSet)[i].input.size()) {
			return (*trainingSet)[i].input[j];
		}
		return 0.0;
	}

	EXPORT_API double getOutput(std::vector<trainingExample> *trainingSet, int i, int j) {
		if (i < trainingSet->size() && j < (*trainingSet)[i].output.size()) {
			return (*trainingSet)[i].output[j];
		}
		return 0.0;
	}

	/* TRAINING SERIES */

	EXPORT_API trainingSeries * createTrainingSeries() {
		return new trainingSeries();
	}

	EXPORT_API void destroyTrainingSeries(trainingSeries *series) {
		delete series;
	}

	/*Adds an array of inputs (one feature) into a series*/
	EXPORT_API void addInputsToSeries(trainingSeries *series, double *inputs, int numInputs) {
		// Define temp vector
		std::vector<double> tempFeatureVector;
		// Add external feature to it
		for (int i = 0; i < numInputs; i++) {
			tempFeatureVector.push_back(inputs[i]);
		}
		// Push temp vector with external feature to series
		series->input.push_back(tempFeatureVector);
	}

	/*Adds the label (output) to a series*/
	EXPORT_API void addLabelToSeries(trainingSeries *series, const char *labelString) {
		series->label = labelString;
	}

	/*Creates a collection of training series/sequences*/
	EXPORT_API std::vector<trainingSeries> * createTrainingSeriesCollection() {
		return new std::vector<trainingSeries>();
	}

	EXPORT_API void destroyTrainingSeriesCollection(std::vector<trainingSeries> *seriesCollection) {
		delete seriesCollection;
	}

	EXPORT_API void addSeriesToSeriesCollection(std::vector<trainingSeries> *seriesCollection , trainingSeries *series) {
		seriesCollection->push_back(*series);
	}

#pragma endregion


#pragma region Training Logic

	EXPORT_API bool trainRegression(regression *model, std::vector<trainingExample> *trainingSet) {
		//model->initialize();
		return  model->train(*trainingSet);
	}

	EXPORT_API bool trainClassification(classification *model, std::vector<trainingExample> *trainingSet) {
		//model->initialize();
		return  model->train(*trainingSet);
	}

	EXPORT_API bool trainSeriesClassification(seriesClassification *model, std::vector<trainingSeries> *trainingSeriesSet) {
		return model->train(*trainingSeriesSet);
	}

#pragma endregion


#pragma region Running Logic

	EXPORT_API int process(modelSet<double> *model, double *input, int numInputs, double *output, int numOutputs) {
		std::vector<double> inputVector;
		for (int i = 0; i < numInputs; i++) {
			inputVector.push_back(input[i]);
		}
		std::vector<double> outputVector = model->run(inputVector);

		if (numOutputs > outputVector.size()) {
			numOutputs = outputVector.size();
		}

		for (int i = 0; i < numOutputs; i++) {
			output[i] = outputVector[i];
		}
		return numOutputs;
	}

	EXPORT_API void resetSeriesClassification(seriesClassification *model) {
		//model->initialize();
		model->reset();
	}

	EXPORT_API const char * runSeriesClassification(seriesClassification *model, trainingSeries *runningSeries) {
		//model->reset();
		//return 0;	
		std::string outputDTW = model->run(runningSeries->input);
		// I need to add an offset to return this to C#
		char * stringToReturn = new char[outputDTW.size() + 1];
		std::copy(outputDTW.begin(), outputDTW.end(), stringToReturn);
		// Add last char to mark the end of the string (I think?)
		stringToReturn[outputDTW.size()] = '\0';
		return stringToReturn;
	}

	EXPORT_API int getSeriesClassificationCosts(seriesClassification *model, double *output, int numOutputs) {
		std::vector<double> outputVector = model->getCosts();

		if (numOutputs > outputVector.size()) {
			numOutputs = outputVector.size();
		}

		for (int i = 0; i < numOutputs; i++) {
			output[i] = outputVector[i];
		}
		return numOutputs;
	}

#pragma endregion


} // end of export C block
