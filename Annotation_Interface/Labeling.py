import requests
import json


class mydict(dict):
    def __str__(self):
        return json.dumps(self)

    def __repr__(self):
        return json.dumps(self)

def extractData(dataDict):
    results = []
    for completion in dataDict["completions"]:
        resultList = completion["result"]
        charSpans = []
        answerTexts = []
        weighting=-1
    
        for result in resultList:
            if result["from_name"]=="answer":
                # With the charspans, the end index is the last character that is inclusive
                charSpans.append((result["value"]["start"], result["value"]["end"]-1))
                answerTexts.append(result["value"]["text"])
            elif  result["from_name"]=="weighting":
                weighting=result["value"]["rating"]
    
        if len(answerTexts)<1:
                return None
    
        resultDict = {}
        resultDict["charSpans"] = charSpans
        resultDict["texts"] = answerTexts
        if weighting>=0:
            resultDict["weighting"] = weighting
        results.append(resultDict)
    return results[0]


def getLabelList(contextAll: list, questionsAll: list, metrics: list, queryIdx: list) -> list:
    """
    Function to build the list for the LabelStudio API from the active learning framework output
    :params: 
        - contextAll: list of all context data from all samples
        - questionsAll: list of all question data from all samples
        - metrics: list of the metrics for the n-queried samples
        - queryIdx: list of indices in regard to all data for the n-queried samples
    :return: labelStructure: 2-dim list with rows as data for a sample containing question, context and metrics
             e.g. [['Context 2', 'Q 2', {'BALD': '5'}], ['Context 3', 'Q 3', {'BALD': '10'}]]
    """ 
    labelStructure = []
    for i, queryIdx in enumerate(queryIdx):
        labelStructure.append([contextAll[queryIdx], questionsAll[queryIdx], {"BALD": str(metrics[i])}])
    return labelStructure


def label(instances):
    """
    Method sends the question and the text for labeling to the annotation framework and returns a list dictionaries with the text and the char spans. If necessary, there is also a weighting.
    Important: The Label-Studio server must be running with the correct configuration (the "questionAnswering" project already has the right one) and for a port other than 8080, the port variable must be changed manually.

    :param instances: List of entities to be labeled, consisting of a list with text, question and a dictionary of metrics e.g.  [[text, query, {"metric_1":4, "metric_2":2}],[text2, query2,{"metric_1":3, "metric_2":3}]
    :return: List of Dictionaries containing a list of char spans and texts. A weighting is optionally specified
             e.g. [{'charSpans': [(92, 125), (156, 168)], 'texts': ['a golden statue of the Virgin Mary', 'Main Building'], 'weighting': 4},...]
    """ 
    
    port = "8080"
    
    
    importList = []
    for instance in instances:
        couples = [["text",instance[0].replace('"','#$@')],
               ["question",instance[1].replace('"','#$@')]]
        
        for key,value in instance[2].items():
            couples.append([key, str(value)])

        pairs =  mydict(couples)
        importList.append(pairs)

    
    headers = {
        'Content-Type': 'application/json',
    }
    data = str(importList)
  
    # Sends the request to Label Studio with the text to be annotated
    response = requests.post('http://localhost:'+port+'/api/project/sendTask', headers=headers, data=data)
    
    if not response.ok:
        print("Something went wrong")
        return None
    
    response = requests.get('http://localhost:'+port+'/api/project/getLabels')

    if not response.ok:
        print("Something went wrong")
        return None

    responseList = json.loads(response.content)
    resultList = []
    for dataDict in responseList:
        resultList.append(extractData(dataDict))
    return resultList