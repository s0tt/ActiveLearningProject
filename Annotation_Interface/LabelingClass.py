import requests
import json

import sys 
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../label-studio/label_studio'))
from server import main as serverStart
import _thread
import time

import xml.etree.ElementTree as ET

class mydict(dict):
    def __str__(self):
        return json.dumps(self)

    def __repr__(self):
        return json.dumps(self)


"""
Data labeling class. 
The Label-Studio Server is started when instantiating. 
Then data can be labeled using the "label" function. 
Important: Do not forget to stop the server using the "stopServer" method or the corresponding button on the Label-Studio interface. 
""" 
class LabelInstance:
  def __init__(self, port, dataPoints, statisticLabels, helpTexts):
    """
    Creates the config file and starts the Label-Studio server.
    
    :param port: Port on which the Label-Studio server should run  
    :param dataPoints: Dictionary made up of the names and the associated data type (e.g. text, image, audio) of the data points. E.g. for question answering {'text': 'Text', 'question': 'Text'}
    :param statisticLabels: This is important for naming the statistics charts. A list must be provided for each diagram. The first value represents the diagram title and the second represents the title of the y-axis.
                            This list must be transferred as a string e.g. '[["Metric_1", "Value"], ["Accuracy", "Value"]]'  
    :param helpTexts: With the specification, help texts for metrics can be provided. It is specified as a dictionary, which takes the title of the respective metric (must be consistent with the metric name when labeling) with its associated help text as a value. 
                      This dictionary must also be passed as a string e.g. '{"metric_1": "info 1", "metric_2": "info 2"}' 
    """
    
    self.port = port
    self.dataPoints=dataPoints
    
    config = self.getConfigQuestionAnswering()
    
    # create a new config file
    configdata = ET.tostring(config, encoding="unicode")
    configfile = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../label-studio/config.xml'), "w")

    configfile.write(configdata)
    configfile.close()
    time.sleep(5)
    self.startServer(statisticLabels, helpTexts)

  def startServer(self, statisticLabels, helpTexts):
    """
    Starts the Label-Studio server 
    """
    #project_name="imageDataLabeling"
    project_name="questionAnswering"

    try:
        _thread.start_new_thread(serverStart, (project_name, self.port))
    except:
       print("Error: unable to start thread")
    time.sleep(10)
    
    # Set filters
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = []
    data.append('{"id":1,"title":"New_Data","ordering":["tasks:id"],"type":"list","target":"tasks","filters":{"conjunction":"and","items":[{"filter":"filter:tasks:completed_at","operator":"empty","value":true,"type":"Datetime"},{"filter":"filter:tasks:data.new","operator":"equal","value":"True","type":"String"}]},"hiddenColumns":{"explore":["tasks:completions_results","tasks:predictions_score","tasks:predictions_results","tasks:data.new","tasks:total_predictions","tasks:cancelled_completions","tasks:total_completions"],"labeling":["tasks:data.text","tasks:data.new","tasks:total_completions","tasks:cancelled_completions","tasks:total_predictions","tasks:completions_results","tasks:predictions_score","tasks:predictions_results"]},"selectedItems":{"all":false,"included":[]},"columnsWidth":{},"columnsDisplayType":{},"gridWidth":4}')
    data.append('{"id":2,"title":"Uncompleted","ordering":["tasks:id"],"type":"list","target":"tasks","filters":{"conjunction":"and","items":[{"filter":"filter:tasks:completed_at","operator":"empty","value":true,"type":"Datetime"}]},"hiddenColumns":{"explore":["tasks:completions_results","tasks:predictions_score","tasks:predictions_results","tasks:total_completions","tasks:cancelled_completions","tasks:total_predictions"],"labeling":["tasks:data.text","tasks:data.new","tasks:total_completions","tasks:cancelled_completions","tasks:total_predictions","tasks:completions_results","tasks:predictions_score","tasks:predictions_results"]},"selectedItems":{"all":false,"included":[]},"columnsWidth":{},"columnsDisplayType":{},"gridWidth":4}')
    data.append('{"id":3,"title":"All","ordering":["tasks:id"],"type":"list","target":"tasks","filters":{"conjunction":"and","items":[]},"hiddenColumns":{"explore":["tasks:completions_results","tasks:predictions_score","tasks:predictions_results"],"labeling":["tasks:data.text","tasks:data.new","tasks:total_completions","tasks:cancelled_completions","tasks:total_predictions","tasks:completions_results","tasks:predictions_score","tasks:predictions_results"]},"selectedItems":{"all":false,"included":[]},"columnsWidth":{},"columnsDisplayType":{},"gridWidth":4}')
    
    for i in range(len(data)):   
        response = requests.post('http://localhost:' + str(self.port) + '/api/project/tabs/'+str(i+1)+'?interaction=filter', headers=headers, data=data[i].encode('utf-8'))
        if not response.ok:
            print("Something went wrong")
    time.sleep(2)

    response = requests.post('http://localhost:' + str(self.port) + '/api/statistics', headers=headers, data=statisticLabels.encode('utf-8'))
    if not response.ok:
        print("Something went wrong")
        
    response = requests.post('http://localhost:' + str(self.port) + '/api/helpTexts', headers=headers, data=helpTexts.encode('utf-8'))
    if not response.ok:
        print("Something went wrong")
    
  def stopServer(self):
    """
    Stops the Label-Studio server 
    """
    
    response = requests.get('http://localhost:' + str(self.port) + '/api/shutdown')

    if not response.ok:
        print("Something went wrong")
        return False
    return True
 
  def getConfigChoices(self):
    dataPoints=self.dataPoints
    
    config = ET.Element('View')
    toName=""
    for name, dataType in dataPoints.items(): 
        dataPoint = ET.SubElement(config, dataType)
        dataPoint.set('name', name)
        dataPoint.set('value','$'+name)
        toName=name
    choices = ET.SubElement(config, 'Choices')
    choices.set('name','choice')
    choices.set('toName',toName)
    choices.set('showInLine','true')
    choice1 = ET.SubElement(choices, 'Choice')
    choice2 = ET.SubElement(choices, 'Choice')
    choice1.set('value','Boeing')
    choice1.set('background','blue')
    choice2.set('value','Airbus')
    choice2.set('background','green')
    text2 = ET.SubElement(config, 'Text')
    text2.set('name','new')
    text2.set('value','$new')
    return config

  def getConfigImageNumber(self):
    dataPoints=self.dataPoints
    
    config = ET.Element('View')
    toName=""
    for name, dataType in dataPoints.items(): 
        dataPoint = ET.SubElement(config, dataType)
        dataPoint.set('name', name)
        dataPoint.set('value','$'+name)
        toName=name
    view = ET.SubElement(config, 'View')
    view.set('showInLine','true')
    textArea = ET.SubElement(view, 'TextArea')
    textArea.set('name','number')
    textArea.set('toName',toName)
    textArea.set('placeholder','number (0-9)')
    textArea.set('editable','true')
    textArea.set('maxSubmissions','1')
    return config

  def getConfigQuestionAnswering(self):
    dataPoints=self.dataPoints
      
    configQuestionAnswering = ET.Element('View')
    configQuestionAnswering.set('style','display: flex;')
    
    view1 = ET.SubElement(configQuestionAnswering, 'View')
    view1.set('style','width: 150px; padding-left: 2em; margin-right: 2em; background: #f1f1f1; border-radius: 3px')
    labels = ET.SubElement(view1, 'Labels')
    labels.set('name','answer')
    labels.set('toName','text')
    label = ET.SubElement(labels, 'Label')
    label.set('value','Answer')
    
    view2 = ET.SubElement(configQuestionAnswering, 'View')
    view3 = ET.SubElement(view2, 'View')
    view3.set('style','width: 100%; display: block')
    header1 = ET.SubElement(view3, 'Header')
    header1.set('value','$question')
    view4 = ET.SubElement(view2, 'View')
    view4.set('style','height: 300px; overflow-y: auto')
    text1 = ET.SubElement(view4, 'Text')
    text1.set('name','text')
    text1.set('value','$text')
    view5 = ET.SubElement(view2, 'View')
    header2 = ET.SubElement(view5, 'Header')
    header2.set('value','Weighting')
    rating = ET.SubElement(view5, 'Rating')
    rating.set('name','weighting')
    rating.set('toName','text')
    text2 = ET.SubElement(view2, 'Text')
    text2.set('name','new')
    text2.set('value','$new')
    return configQuestionAnswering
    
  def label(self, instances, statisticData):
    """
    Method for labeling data which belong to the specified config and data types. 
    
    :param instances: List of entities to be labeled, consisting of the dataPoints and a dictionary of metrics e.g. for question answering [[text, query, {"metric_1":4, "metric_2":2}],[text2, query2,{"metric_1":3, "metric_2":3}]
    :param statisticData: List with the new statistic numbers for the diagrams. The order is decisive for the assignment to the diagram title specified during initialization. e.g. [0.4, 0.7]
    :return: Dictionary, which contains the data provided by the user and the originally requested data (key "data"). 
             e.g. for question answering [{'charSpans': [(92, 125), (156, 168)], 'texts': ['a golden statue of the Virgin Mary', 'Main Building'], 'weighting': 4 , 'data': {'metric_1': '2.3', 'new': False, 'question': 'query', 'metric_2': '23', 'text': 'text'},...]
    """ 
    
    importList = []
    for instance in instances:
        
        index = 0
        couples = []
        for name, dataType in self.dataPoints.items(): 
            if dataType=="Text":
                couples.append([name, instance[index].replace('"','#$@')])
            else:
                fileName = instance[index].split("/")[-1]
                filePath = instance[index].split("/"+fileName)[0]
                couples.append([name, "/data/" + fileName + "?d=" + filePath])
            index+=1

        for key,value in instance[-1].items():
            couples.append([key, str(value)])

        pairs =  mydict(couples)
        importList.append(pairs)

    data = str(importList)

    responseList = self.restInteraction(data, str(statisticData))
    resultList = []
    for dataDict in responseList:
        resultList.append(self.extractData(dataDict))
    return resultList

  def extractData(self, dataDict):
    """
    The method extracts the necessary information for each data point from the data exported by Label-Studio.
    
    :param dataDict: Dictionary that contains the data imported from the Label-Studio for a specific data point  
    :return: Dictionary containing the extracted data 
    """
    
    results = []
    for completion in dataDict["completions"]:
        resultList = completion["result"]
        resultDataDict = {}
    
        for result in resultList:
            if result["from_name"]=="answer":
                if "charSpans" not in resultDataDict:
                    # With the charspans, the end index is the last character that is inclusive
                    resultDataDict["charSpans"]=[(result["value"]["start"], result["value"]["end"]-1)]
                else:
                    resultDataDict["charSpans"].append((result["value"]["start"], result["value"]["end"]-1))
                    
                if "texts" not in resultDataDict:
                    resultDataDict["texts"]=[result["value"]["text"]]
                else:
                    resultDataDict["texts"].append(result["value"]["text"])
            elif result["from_name"]=="weighting":
                weighting=result["value"]["rating"]
                
                resultDataDict["weighting"]=result["value"]["rating"]
                
            elif result["from_name"]=="choice":
                resultDataDict["choice"]=result["value"]["choices"]
                
            elif result["from_name"]=="number":
                resultDataDict["choice"]=result["value"]["text"][0]
                
        resultDataDict["data"]=dataDict["data"]
        results.append(resultDataDict)
    return results[-1]
    
  def restInteraction(self, data, statisticData):
    """
    The method sends data to the Label-Studio server that are to be labeled by the user and receives the labeled data.
    
    :param data: List as a string that contains the data records 
    :param statisticData: List with the new statistic numbers for the diagrams. The order is decisive for the assignment to the diagram title specified during initialization. List must be passed as a string.
    :return: List with the labeled data 
    """ 
    
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post('http://localhost:' + str(self.port) + '/api/project/sendTask', headers=headers, data=data.encode('utf-8'))
    
    if not response.ok:
        print("Something went wrong")
        return None

    response = requests.post('http://localhost:' + str(self.port) + '/api/statistics', headers=headers, data=statisticData.encode('utf-8'))
    if not response.ok:
        print("Something went wrong")
    
    response = requests.get('http://localhost:' + str(self.port) + '/api/project/getLabels')

    if not response.ok:
        print("Something went wrong")
        return None

    responseList = json.loads(response.content)
    return responseList

"""
e.g.
inputList = ["C:/Bilder/test.jpg", "C:/Bilder/test2.jpg", "C:/Bilder/test3.jpg"]
image = [[inputList[0], {"metric_1":14, "metric_2":1.2}],[inputList[1],{"metric_1":23, "metric_2":2.3}],[inputList[2], {"metric_1":8, "metric_2":7}]]
p1 = LabelInstance(8080, {'image':'Image'}, '[["Mean Metric", "Value"], ["Accuracy", "Value"]]', '{"metric_1":"info 1", "metric_2":"info 2"}')
print(p1.label(image, [0.4,0.7]))
p1.stopServer()
"""