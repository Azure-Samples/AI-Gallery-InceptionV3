import inception
import json
from azureml.core.model import Model

def init():
    global node_lookup

    # Creates graph from saved graph_def.pb.
    inception.create_graph(Model.get_model_path('classify_image_graph_def'))
    node_lookup = inception.NodeLookup(
        Model.get_model_path('imagenet_label_map'),
        Model.get_model_path('imagenet_s2h_label_map'))

def run(raw_data):
    try:
        res = inception.run_inference_on_image(node_lookup, raw_data)
        return res

    except Exception as e:
        return str(e)
