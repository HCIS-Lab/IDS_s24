import json

f = open('eval_scenarios.json')
eval_scenarios = json.load(f)

content = []
for town in eval_scenarios['available_scenarios'][0]:
  content.append(f"{town}\n")
  for scenario_type in eval_scenarios['available_scenarios'][0][town]:
    if scenario_type['scenario_type'][-1] in ['1', '2', '3']:
      continue
    else:
      content.append(f"{scenario_type['scenario_type']}\n")
      for scenario_config in scenario_type['available_event_configurations']:
        x = scenario_config['transform']['x']
        y = scenario_config['transform']['y']
        content.append(f"{x}, {y}\n")

with open('interaction_scenarios.txt', 'w') as file:
  file.writelines(content)
        