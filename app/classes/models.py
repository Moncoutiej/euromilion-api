from typing import List, Optional
from pydantic import BaseModel, Field    
import datetime

class DrawProba(BaseModel):
    """The Draw win and lose probabilities,  
    """
    win: float
    lose: float
    
class ResponseJson(BaseModel):
    """Default Response
    """
    message: str = "OK"
    status_code: int = 200

class Metric(BaseModel):
    """Metric Object containing information about a metric
    """
    metric_name: str
    value: float

    
class MLModel(BaseModel):
    """Machine Learning Model Object
    """
    metrics: List[Metric]
    model_name: str
    training_params : dict
    

class DataLine(BaseModel):
    """Line Data Object with the appropriate dataset format
    """
    date: str = Field(
        str(datetime.date.today()), title="Date of the draw, Format : YYYY-MM-DD", max_length=10
    )
    n1: int = Field(
        ..., title="The first number of the draw", gt=0, lt=51
    )
    n2: int = Field(
        ..., title="The second number of the draw", gt=0, lt=51
    )
    n3: int = Field(
        ..., title="The third number of the draw", gt=0, lt=51
    )
    n4: int = Field(
        ..., title="The forth number of the draw", gt=0, lt=51
    )
    n5: int = Field(
        ..., title="The fifth number of the draw", gt=0, lt=51
    )
    e1: int = Field(
        ..., title="The first star number of the draw", gt=0, lt=13
    )
    e2: int = Field(
        ..., title="The second star number of the draw", gt=0, lt=13
    )
    winner: Optional[int] = Field(
        None, title="The number of winner for this draw", ge=0
    )
    gain: Optional[int] = Field(
        None, title="The amount of monney for the winner of the draw", ge=0
    )