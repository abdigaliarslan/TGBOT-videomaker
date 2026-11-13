"""баланс в токенах: ролик = 23 токена, цена задачи, оплаты в токенах

Revision ID: 9b4e2f7a1c35
Revises: 3f1c8a9d2b6e
Create Date: 2026-11-13 18:00:00

Всё, что считалось роликами, пересчитывается по 23 токена за ролик (цена
ролика на момент перехода, `Settings.tokens_per_video`):

* `users.balance` — роликов → токенов;
* `users.ref_credit` — сотые ролика → сотые токена; целые токены из копилки
  сразу уходят на баланс, как это делает `repo.credit_payment`;
* `payments.videos` → `payments.tokens`, старые строки × 23;
* `payments.ref_bonus` — сотые ролика → сотые токена;
* `jobs.cost` — новая колонка: у задачи, за которую списан и не возвращён
  ролик, цена 23 токена — её вернёт возврат, если монтаж ещё упадёт.

Никто ничего не теряет: на каждый ролик баланса приходится ровно цена ролика.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9b4e2f7a1c35'
down_revision: Union[str, None] = '3f1c8a9d2b6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Обязан совпадать с `Settings.tokens_per_video` на момент выкладки: иначе
# ролик на балансе стоил бы не столько токенов, сколько с него спишут.
RATE = 23


def upgrade() -> None:
    op.add_column('jobs', sa.Column('cost', sa.Integer(), server_default='0', nullable=False))
    op.execute(f"UPDATE jobs SET cost = {RATE} WHERE charged")
    op.execute(
        f"UPDATE users SET balance = balance * {RATE} + (ref_credit * {RATE}) / 100, "
        f"ref_credit = (ref_credit * {RATE}) % 100"
    )
    op.alter_column('payments', 'videos', new_column_name='tokens')
    op.execute(f"UPDATE payments SET tokens = tokens * {RATE}, ref_bonus = ref_bonus * {RATE}")


def downgrade() -> None:
    # Обратно — с округлением вниз: дробь ролика возвращается в копилку.
    op.execute(f"UPDATE payments SET tokens = tokens / {RATE}, ref_bonus = ref_bonus / {RATE}")
    op.alter_column('payments', 'tokens', new_column_name='videos')
    op.execute(
        f"UPDATE users SET ref_credit = ((balance % {RATE}) * 100 + ref_credit) / {RATE}, "
        f"balance = balance / {RATE}"
    )
    op.drop_column('jobs', 'cost')
