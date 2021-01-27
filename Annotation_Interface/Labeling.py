import requests
import json


def label(question, text):
    """
    Method sends the question and the text for labeling to the annotation framework and returns a dictionary with the text and the char spans. If necessary, there is also a weighting.
    Important: The Label-Studio server must be running with the correct configuration (the "questionAnswering" project already has the right one) and for a port other than 8080, the port variable must be changed manually.

    :param question: Question which should be answered
    :param text: Text from which the answer to the question should be extracted
    :return: Dictionary containing a list of char spans and texts. A weighting is optionally specified
             e.g. {'charSpans': [(92, 125), (156, 168)], 'texts': ['a golden statue of the Virgin Mary', 'Main Building'], 'weighting': 4}
    """ 
    
    port = "8080"

    # Quotation marks are misinterpreted in request
    editText = text.replace('"','#$@')
    
    headers = {
        'Content-Type': 'application/json',
    }
    data = '[{"text": "' + editText + '", "question": "' + question + '"}]'
    
    # Sends the request to Label Studio with the text to be annotated
    response = requests.post('http://localhost:'+port+'/api/project/sendTask', headers=headers, data=data.encode('utf-8'))
    
    if not response.ok:
        print("Something went wrong")
        return None
    
    response = requests.get('http://localhost:'+port+'/api/project/getLabel')

    if not response.ok:
        print("Something went wrong")
        return None

    responseDict = json.loads(response.content)
    resultList = responseDict["completions"][0]["result"]
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

    return resultDict