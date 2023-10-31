from typing import Dict, List
import pandas as pd
import time
from constants import all_skill, passive_skill, evolve_requirement, active_skill

# def get_skill_list() -> List[str]:
#     df = pd.read_csv('skill_data.csv')
#     skills = list(df['skills'])
    
#     all_skill = []
#     for s in skills:
#         skills_ind = s.split(',')
#         skills_ind = [sk.split('_')[-1] for sk in skills_ind if sk.split('_')[-1] != ""]
        
#         all_skill += skills_ind
#     all_skill = list(set(all_skill))
    
#     return all_skill


def transform_skill_list(skill_list) -> Dict[str, any]:
    skill_dict = {}
    for skill in skill_list:
        if skill in skill_dict:
            skill_dict[skill] += 1
        else:
            skill_dict[skill] = 1
    
    return skill_dict


def sort_dict_by_value(my_dict):
    sorted_tuples = sorted(my_dict.items(), key=lambda item: item[1])
    sorted_dict = dict(sorted_tuples)
    return sorted_dict


def get_skills_to_skip(skill_list: List[str]) -> List[str]:
    
    skill_dict = transform_skill_list(skill_list)
    
    skills_to_skip = ["end"]
    active_list = {}
    passive_list = {}
    
    for skill, level in skill_dict.items():
        if skill in passive_skill:
            passive_list[skill] = level
            if level == 5:
                skills_to_skip.append(skill)
        else:
            active_list[skill] = level
            if level == 6:
                skills_to_skip.append(skill)
                
            if level == 5:
                required_skill = evolve_requirement[skill]
                if skill == 'Black Hawk':
                    skills_to_skip.append(skill)
                elif skill == 'White Eagle':
                    if skill_list.count(required_skill) != 5:
                        skills_to_skip.append(skill)    
                elif required_skill not in skill_dict:
                    skills_to_skip.append(skill)
                    
                    
    if len(active_list) == 6:
        to_add = [s for s in active_skill if s not in active_list]
        skills_to_skip += to_add
    if len(passive_list) == 6:
        to_add = [s for s in passive_skill if s not in passive_list]
        skills_to_skip += to_add
        
    skills_to_skip = list(set(skills_to_skip))
    
    return skills_to_skip