"""jobs.top_source_key — второй исходник режима «Два видео»

Revision ID: 3f1c8a9d2b6e
Revises: 5c0d2a7e9b14
Create Date: 2026-11-05 12:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3f1c8a9d2b6e'
down_revision: Union[str, None] = '5c0d2a7e9b14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('top_source_key', sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'top_source_key')
