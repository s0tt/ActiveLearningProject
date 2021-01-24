import requests
from io import BytesIO
import csv
import codecs
import time
import json
import webbrowser

from zipfile import ZipFile


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
    response = requests.post('http://localhost:'+port+'/api/project/import', headers=headers, data=data)
    
    # Just an interim solution until the label page updates automatically.
    url = 'http://localhost:'+port+'/tasks?tab=1&labeling=1'
    webbrowser.open_new(url) # opens in default browser
    
    if not response.ok:
        print("Something went wrong")
        return None
    
    params = (
        ('format', 'CSV'),
    )

    response = requests.get('http://localhost:'+port+'/api/project/getLabel', params=params)
    if not response.ok:
        print("Something went wrong")
        return None
    
    # The answer is a CSV file in a ZIP archive, that is read in and data is extracted
    zip_file = ZipFile(BytesIO(response.content))
    files = zip_file.namelist()
    with zip_file.open(files[0], 'r') as csvfile:
        csvreader = csv.reader(codecs.iterdecode(csvfile, 'utf-8'))
        header = next(csvreader)
        
        if "answer" not in header:
            return None

        answerIndex = header.index("answer")
        
        exportString = next(csvreader)
        answerList = exportString[answerIndex][1:-1].replace("},","};").split(";")
        charSpans = []
        answerTexts = []
        for answer in answerList:
            answerDict = json.loads(answer)
            # With the charspans, the end index is the last character that is inclusive
            charSpans.append((answerDict["start"], answerDict["end"]-1))
            answerTexts.append(answerDict["text"])
        resultDict = {}
        resultDict["charSpans"] = charSpans
        resultDict["texts"] = answerTexts
        if "weighting" in header:
            weightingIndex = header.index("weighting")
            weightingDict = json.loads(exportString[weightingIndex][1:-1])
            resultDict["weighting"] = weightingDict["rating"]
        return resultDict