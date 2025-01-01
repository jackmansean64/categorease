from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, TypeAdapter

class Transaction(BaseModel):
    date: datetime = Field(alias="Date")
    description: str = Field(alias="Description")
    category: Optional[str] = Field(alias="Category", default=None)
    amount: float = Field(alias="Amount")
    labels: Optional[str] = Field(alias="Labels", default=None)
    notes: Optional[str] = Field(alias="Notes", default=None)
    account: Optional[str] = Field(alias="Account", default=None)
    account_number: Optional[str] = Field(alias="Account #", default=None)
    institution: Optional[str] = Field(alias="Institution", default=None)
    month: Optional[datetime] = Field(alias="Month", default=None)
    week: Optional[datetime] = Field(alias="Week", default=None)
    transaction_id: Optional[str] = Field(alias="Transaction ID", default=None)
    account_id: Optional[str] = Field(alias="Account ID", default=None)
    check_number: Optional[str] = Field(alias="Check Number", default=None)
    full_description: Optional[str] = Field(alias="Full Description", default=None)
    date_added: Optional[datetime] = Field(alias="Date Added", default=None)


class Category(BaseModel):
    category: str = Field(alias="Category")
    group: Optional[str] = Field(alias="Group", default=None)
    type: Optional[str] = Field(alias="Type", default=None)


class CategorizedTransaction(BaseModel):
    transaction_id: str
    date: datetime
    description: str
    category: str