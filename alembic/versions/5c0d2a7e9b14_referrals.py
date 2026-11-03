"""Приглашения: кто кого привёл, копилка долей ролика, бонус с оплаты

Revision ID: 5c0d2a7e9b14
Revises: 7ea1fa6e2059
Create Date: 2026-11-03 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '5c0d2a7e9b14'
down_revision: Union[str, None] = '7ea1fa6e2059'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('referrer_id', sa.BigInteger(), nullable=True))
    op.add_column('users', sa.Column('referred_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('users', sa.Column('ref_credit', sa.Integer(), server_default='0', nullable=False))
    op.create_foreign_key(
        'users_referrer_id_fkey', 'users', 'users', ['referrer_id'], ['id'], ondelete='SET NULL'
    )
    op.create_index(op.f('ix_users_referrer_id'), 'users', ['referrer_id'], unique=False)
    op.add_column('payments', sa.Column('ref_bonus', sa.Integer(), server_default='0', nullable=False))


def downgrade() -> None:
    op.drop_column('payments', 'ref_bonus')
    op.drop_index(op.f('ix_users_referrer_id'), table_name='users')
    op.drop_constraint('users_referrer_id_fkey', 'users', type_='foreignkey')
    op.drop_column('users', 'ref_credit')
    op.drop_column('users', 'referred_at')
    op.drop_column('users', 'referrer_id')
