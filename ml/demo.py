"""Stable synthetic demo records for application exploration and injected-outlier tests."""
import random


def demo_rows():
    rng = random.Random(42)
    rows = []
    for month in range(1,13):
        rows.append({'id':f'demo-income-{month}','date':f'2025-{month:02d}-01',
                     'description':'Salary payroll','amount':50000,'category':'Income'})
        for index in range(10):
            category,description = [('Groceries','Supermarket grocery'),('Restaurants','Cafe lunch'),
                                    ('Transportation','Metro transport')][index%3]
            rows.append({'id':f'demo-{month}-{index}','date':f'2025-{month:02d}-{index+2:02d}',
                         'description':description,'amount':-round(rng.uniform(150,900),2),
                         'category':'Other' if month==1 and index<3 else category})
    rows.append({'id':'demo-outlier','date':'2025-12-20','description':'Unusually large cafe payment',
                 'amount':-24000,'category':'Restaurants'})
    return rows
