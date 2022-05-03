import json
import os 
os.system("pip install numpy tensorflow crcmod")
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict


import crcmod

def _masked_crc32c(value):
    crc = crcmod.predefined.mkPredefinedCrcFun("crc-32c")(value)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

def read_tfrecords(tfrecords):
    import io
    import struct
    tfrecords_bytes = io.BytesIO(tfrecords)

    examples = []

    while True:
        length_header = 12
        buf = tfrecords_bytes.read(length_header)
        if not buf:
            # reached end of tfrecord buffer, return examples
            return examples

        if len(buf) != length_header:
            raise ValueError("TFrecord is fewer than %d bytes" % length_header)
        length, length_mask = struct.unpack("<QI", buf)
        length_mask_actual = _masked_crc32c(buf[:8])
        if length_mask_actual != length_mask:
            raise ValueError("TFRecord does not contain a valid length mask")

        length_data = length + 4
        buf = tfrecords_bytes.read(length_data)
        if len(buf) != length_data:
            raise ValueError("TFRecord data payload has fewer bytes than specified in header")
        data, data_mask_expected = struct.unpack("<%dsI" % length, buf)
        data_mask_actual = _masked_crc32c(data)
        if data_mask_actual != data_mask_expected:
            raise ValueError("TFRecord has an invalid data crc32c")

        # Deserialize the tf.Example proto
        example = tf.train.Example()
        example.ParseFromString(data)
        example_features = MessageToDict(example)['features']['feature']['features']['floatList']['value']
        # Extract a feature map from the example object
        examples.append(example_features)
        
    return examples

def read_csv(csv):
      return np.array([[float(j) for j in i.split(',')] for i in csv.splitlines()])


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    
    if context.request_content_type == 'text/csv':

        payload = data.read().decode("utf-8")
        inputs = read_csv(payload)
      
        input = {
            'inputs': inputs.tolist()
            }
        
     
        return json.dumps(input)
    
    if context.request_content_type == "application/x-tfrecord":
    
        payload = data.read()
        examples = read_tfrecords(payload)
        
        input = {
            'inputs': examples
            }
        
   
        return json.dumps(input)
        

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
            context.request_content_type or "unknown"))

    
def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """

    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type

