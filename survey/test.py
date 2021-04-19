#import torch
import pickle

# def transform_tensors(data):
#     for key, value in data.items():
#         if torch.is_tensor(data[key]):
#             data[key] = data[key].detach().numpy()
#     return data

# def transform_if_tensor(data):
#     return data.detach().numpy() if torch.is_tensor(data) else data

def write_new_dict(data):
    new_dict = {}
    new_dict["label"] = transform_if_tensor(data["label"])
    new_dict["metadata"] = {}
    new_dict["metadata"]["answers_per_instance"] = transform_if_tensor(data["metadata"]["answers_per_instance"])
    new_dict["metadata"]["question"] = transform_if_tensor(data["metadata"]["question"])
    new_dict["metadata"]["context"] = transform_if_tensor(data["metadata"]["context"])
    return new_dict



# batch = pickle.load(open("batch_survey.pkl", "rb"))
# multi_instance = pickle.load(open("multi_instance.pkl", "rb"))
# multi_metric = pickle.load(open("multi_metric.pkl", "rb"))

# batch = transform_tensors(batch)
# batch["metadata"] = transform_tensors(batch["metadata"])
# multi_instance = transform_tensors(multi_instance)
# multi_metric = transform_tensors(multi_metric)

# pickle.dump(batch, open("batch_survey_py.pkl", "wb"))
# pickle.dump(multi_instance, open("multi_instance_py.pkl", "wb"))
# pickle.dump(multi_metric, open("multi_metric_py.pkl", "wb"))


# batch = pickle.load(open("batch_survey.pkl", "rb"))
# batch = transform_tensors(batch)
# batch_new = write_new_dict(batch)
# pickle.dump(batch_new, open("batch_survey_py.pkl", "wb"))

batch = pickle.load(open("batch_survey_py.pkl", "rb"))
multi_instance = pickle.load(open("multi_instance_py.pkl", "rb"))
multi_metric = pickle.load(open("multi_metric_py.pkl", "rb"))

