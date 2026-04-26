from pydantic import BaseModel


class IngestResponse(BaseModel):
    status:          str
    source:          str
    parents_stored:  int
    children_stored: int
    dense_dim:       int
    sparse_avg_nnz:  int