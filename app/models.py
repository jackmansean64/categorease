from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    date: datetime = Field(alias="Date")
    description: str = Field(alias="Description")
    category: Optional[str] = Field(alias="Category", default=None)
    amount: float = Field(alias="Amount")
    labels: Optional[str] = Field(alias="Labels", default=None, exclude=True)
    notes: Optional[str] = Field(alias="Notes", default=None, exclude=True)
    account: Optional[str] = Field(alias="Account", default=None)
    account_number: Optional[str] = Field(alias="Account #", default=None, exclude=True)
    institution: Optional[str] = Field(alias="Institution", default=None, exclude=True)
    month: Optional[datetime] = Field(alias="Month", default=None, exclude=True)
    week: Optional[datetime] = Field(alias="Week", default=None, exclude=True)
    transaction_id: Optional[str] = Field(alias="Transaction ID", default=None)
    account_id: Optional[str] = Field(alias="Account ID", default=None, exclude=True)
    check_number: Optional[str] = Field(alias="Check Number", default=None, exclude=True)
    full_description: Optional[str] = Field(alias="Full Description", default=None, exclude=True)
    date_added: Optional[datetime] = Field(alias="Date Added", default=None, exclude=True)

    # noinspection PyNestedDecorators
    @field_validator(
        "description",
        "category",
        "labels",
        "notes",
        "account",
        "account_number",
        "institution",
        "transaction_id",
        "account_id",
        "check_number",
        "full_description",
        mode="before"
    )
    @classmethod
    def coerce_to_string(cls, value):
        if value is None:
            return value
        return str(value)


class Category(BaseModel):
    category: str = Field(alias="Category")
    group: Optional[str] = Field(alias="Group", default=None)
    type: Optional[str] = Field(alias="Type", default=None)


class CategorizedTransaction(BaseModel):
    transaction_id: str
    date: datetime
    description: str
    category: str
