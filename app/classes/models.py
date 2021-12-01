from pydantic import BaseModel

class Metric:
    """ Metric Object containing information
    """
    metric_name: str

class Params:
    """Parameters Object containing informations according to the Machine Learning Model
    """
    depth: int
    
class MLModel(BaseModel):
    """Machine Learning Model Object

    Args:
        BaseModel (class): Base Model from pylantic
    """
    metric: str
    name: str
    params : str