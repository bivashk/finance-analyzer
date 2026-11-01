"""Synthetic behavioral cohort for demonstration, not evidence of customer outcomes."""
import random


def customers():
    rng = random.Random(42)
    result = []
    for person in range(36):
        kind = person%4
        rows = []
        for month in range(1,7):
            salary = [60000,35000,50000,45000][kind]+rng.randint(-1000,1000)
            rows.append({'date':f'2025-{month:02d}-01','description':'Salary payroll','amount':salary,'category':'Income'})
            rows.append({'date':f'2025-{month:02d}-02','description':'Monthly rent','amount':-[12000,17000,10000,11000][kind],'category':'Rent'})
            for purchase in range(8+kind*2):
                base = [500,1700,1200,650][kind]
                amount = base*rng.uniform(.6,1.4)*(2 if kind==2 and month%2 else 1)
                category = ('Shopping' if kind==3 else 'Groceries') if purchase%2 else 'Restaurants'
                rows.append({'date':f'2025-{month:02d}-{purchase+3:02d}','description':f'{category} purchase {purchase}',
                             'amount':-round(amount,2),'category':category})
        result.append({'customer_id':f'demo-customer-{person+1:02d}','transactions':rows})
    return result


if __name__=='__main__':
    import json
    from pathlib import Path
    from backend.customer import segment
    report = segment(customers())
    report['provenance'] = '36 synthetic customer histories; no real customer or business-outcome validation.'
    Path('reports/customer_segmentation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['chosen_k','silhouette_score','mean_seed_stability_adjusted_rand']},indent=2))
