from sqlmodel import SQLModel


class PQ7Request(SQLModel):
    file_url: str


class PQ7Response(SQLModel):
    receipt_number: str
    destination_country: str
    transportation_mode: str
    total_weight: str
    number_of_boxes: int
    export_date: str
