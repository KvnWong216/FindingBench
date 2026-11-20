"""Read-only schema for archived v1 authoring evidence; never an agent API."""
from typing import Annotated, Literal, Union
from pydantic import Field
from rummagebench.core.public_types import MoveAction, TurnAction, PointAction, ReportDoneAction

PUBLIC_SKILLS = ['MOVE','TURN','OPEN','CLOSE','GRASP','PLACE','OBSERVE','REPORT_DONE']
class LegacyObserveAction(PointAction):
    skill: Literal['OBSERVE']
PublicAction = Annotated[Union[MoveAction,TurnAction,PointAction,LegacyObserveAction,ReportDoneAction],Field(discriminator='skill')]
