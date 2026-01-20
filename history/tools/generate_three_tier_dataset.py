#!/usr/bin/env python3
"""
Generate a large three-tier dataset for ByteLogic training
"""

import json
import random
from typing import List, Dict, Any

def generate_family_relationship_examples() -> List[Dict[str, Any]]:
    """Generate family relationship examples."""
    examples = []
    
    names = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry", "irene", "jack", 
             "karen", "leo", "mary", "nick", "olivia", "peter", "quinn", "rachel", "steve", "tina"]
    
    for i in range(25):
        name1, name2, name3 = random.sample(names, 3)
        
        example = {
            "id": f"family_{i+1}",
            "user_query": f"Who are {name1}'s children?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for {name1}'s children
- Need to find all people who have {name1} as their parent
- This is a direct lookup in a parent-child relationship

APPROACH STRATEGY:
- Store parent-child relationships
- Query for all children where {name1} is the parent
- This is a basic fact lookup operation

CONNECTION TO BYTELOGIC:
- Need parent relation to store relationships
- Need query to extract specific results
</thinking>

<pseudocode>
; Problem: Find children of a specific person
; Input: Parent name ({name1})
; Goal: Find all children of {name1}

REL parent

; Facts: Parent-child relationships
FACT parent {name1} {name2}
FACT parent {name1} {name3}
FACT parent {name2} steve

; Rules: None needed for direct lookup

SOLVE
QUERY parent {name1} ?
</pseudocode>

<computation>
REL parent

FACT parent {name1} {name2}
FACT parent {name1} {name3}
FACT parent {name2} steve

SOLVE
QUERY parent {name1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_grandparent_examples() -> List[Dict[str, Any]]:
    """Generate grandparent relationship examples."""
    examples = []
    
    names = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry", "irene", "jack", 
             "karen", "leo", "mary", "nick", "olivia", "peter", "quinn", "rachel", "steve", "tina"]
    
    for i in range(25):
        name1, name2, name3 = random.sample(names, 3)
        
        example = {
            "id": f"grandparent_{i+1}",
            "user_query": f"Who are {name1}'s grandchildren?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for {name1}'s grandchildren
- This requires finding children of {name1}'s children
- Need a transitive relationship: parent of parent
- This is a classic transitive closure problem

APPROACH STRATEGY:
- First find {name1}'s children
- Then find children of those children
- Need to create a derived relationship
- This requires rule-based inference

CONNECTION TO BYTELOGIC:
- Need parent relation for base facts
- Need grandparent relation for derived facts
- Need rule to create transitive relationship
</thinking>

<pseudocode>
; Problem: Find grandchildren of a specific person
; Input: Grandparent name ({name1})
; Goal: Find all grandchildren of {name1}

REL parent
REL grandparent

; Facts: Direct parent-child relationships
FACT parent {name1} {name2}
FACT parent {name2} {name3}
FACT parent {name1} grace
FACT parent grace henry

; Rules: Create grandparent-grandchild relationship
; Base case: Direct parent-child
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2

SOLVE
QUERY grandparent {name1} ?
</pseudocode>

<computation>
REL parent
REL grandparent

FACT parent {name1} {name2}
FACT parent {name2} {name3}
FACT parent {name1} grace
FACT parent grace henry

RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2

SOLVE
QUERY grandparent {name1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_friendship_examples() -> List[Dict[str, Any]]:
    """Generate friendship network examples."""
    examples = []
    
    names = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry", "irene", "jack", 
             "karen", "leo", "mary", "nick", "olivia", "peter", "quinn", "rachel", "steve", "tina"]
    
    for i in range(25):
        name1, name2, name3 = random.sample(names, 3)
        
        example = {
            "id": f"friendship_{i+1}",
            "user_query": f"Who are friends of {name1}'s friends?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for friends of {name1}'s friends (friends-of-friends)
- This is a graph traversal problem
- Need to find connections through one intermediate person
- This requires finding mutual connections

APPROACH STRATEGY:
- Find {name1}'s direct friends
- For each friend, find their friends
- Exclude {name1} herself from the results
- This requires joining the friend relation with itself

CONNECTION TO BYTELOGIC:
- Need friend relation to store connections
- Need rule to find friend-of-friend relationships
- Need query to extract specific results
</thinking>

<pseudocode>
; Problem: Find friends of friends (foaf) for a specific person
; Input: Person name ({name1})
; Goal: Find all people who are friends of {name1}'s friends

REL friend
REL friend_of_friend

; Facts: Direct friend relationships
FACT friend {name1} {name2}
FACT friend {name2} {name3}
FACT friend {name1} grace
FACT friend grace henry

; Rules: Find friend of friend relationships
; Connect through intermediate person
RULE friend_of_friend: SCAN friend MATCH $0, JOIN friend $0, EMIT friend_of_friend $1 $2

SOLVE
QUERY friend_of_friend {name1} ?
</pseudocode>

<computation>
REL friend

FACT friend {name1} {name2}
FACT friend {name2} {name3}
FACT friend {name1} grace
FACT friend grace henry

RULE friend_of_friend: SCAN friend MATCH $0, JOIN friend $0, EMIT friend_of_friend $1 $2

SOLVE
QUERY friend_of_friend {name1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_workplace_examples() -> List[Dict[str, Any]]:
    """Generate workplace relationship examples."""
    examples = []
    
    names = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry", "irene", "jack", 
             "karen", "leo", "mary", "nick", "olivia", "peter", "quinn", "rachel", "steve", "tina"]
    companies = ["techcorp", "innovate", "solutions", "enterprises", "systems", "labs", "ventures", "partners"]
    
    for i in range(25):
        name1, name2, name3 = random.sample(names, 3)
        company = random.choice(companies)
        
        example = {
            "id": f"workplace_{i+1}",
            "user_query": f"Who reports to the same manager as {name1}?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for people who report to {name1}'s manager(s)
- This is a co-worker relationship
- Need to find {name1}'s manager, then others who report to the same manager
- This is a common supervisor problem

APPROACH STRATEGY:
- Find {name1}'s manager(s)
- Find all people who report to those same managers
- Exclude {name1} from the results
- This requires finding common reporting relationships

CONNECTION TO BYTELOGIC:
- Need reports_to relation to store reporting facts
- Need coworker relation for derived facts
- Need rule to find people with common managers
</thinking>

<pseudocode>
; Problem: Find coworkers of a specific employee
; Input: Employee name ({name1})
; Goal: Find all employees who report to the same manager as {name1}

REL reports_to
REL coworker

; Facts: Reporting relationships
FACT reports_to {name1} {company}_manager
FACT reports_to {name2} {company}_manager
FACT reports_to {name3} {company}_manager
FACT reports_to grace ceo

; Rules: Find employees with common managers
RULE coworker: SCAN reports_to MATCH $1, JOIN reports_to $1, EMIT coworker $0 $2

SOLVE
QUERY coworker {name1} ?
</pseudocode>

<computation>
REL reports_to
REL coworker

FACT reports_to {name1} {company}_manager
FACT reports_to {name2} {company}_manager
FACT reports_to {name3} {company}_manager
FACT reports_to grace ceo

RULE coworker: SCAN reports_to MATCH $1, JOIN reports_to $1, EMIT coworker $0 $2

SOLVE
QUERY coworker {name1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_animal_capability_examples() -> List[Dict[str, Any]]:
    """Generate animal capability examples."""
    examples = []
    
    animals = ["eagle", "sparrow", "ostrich", "fish", "shark", "dolphin", "bat", "penguin", "hawk", "owl"]
    capabilities = ["fly", "swim", "run", "jump", "climb", "dig", "crawl", "walk"]
    
    for i in range(25):
        animal1, animal2 = random.sample(animals, 2)
        capability = random.choice(capabilities)
        
        example = {
            "id": f"animal_{i+1}",
            "user_query": f"What animals can {capability}?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for animals with the {capability} capability
- Need to connect animals to their capabilities
- This is a capability lookup problem

APPROACH STRATEGY:
- Store animal-capability relationships
- Query for all animals with {capability} capability
- This is a reverse lookup operation

CONNECTION TO BYTELOGIC:
- Need has_capability relation to store capability facts
- Need query to extract animals by capability
</thinking>

<pseudocode>
; Problem: Find all animals with a specific capability
; Input: Capability ({capability})
; Goal: Find all animals that can {capability}

REL has_capability

; Facts: Animal-capability relationships
FACT has_capability {animal1} {capability}
FACT has_capability {animal2} {capability}
FACT has_capability fish swim
FACT has_capability shark swim

; Rules: None needed for direct lookup

SOLVE
QUERY has_capability ? {capability}
</pseudocode>

<computation>
REL has_capability

FACT has_capability {animal1} {capability}
FACT has_capability {animal2} {capability}
FACT has_capability fish swim
FACT has_capability shark swim

SOLVE
QUERY has_capability ? {capability}
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_course_prerequisite_examples() -> List[Dict[str, Any]]:
    """Generate course prerequisite examples."""
    examples = []
    
    courses = ["cs101", "cs201", "cs301", "math101", "math201", "eng101", "phy101", "bio101", "chem101", "cs150"]
    
    for i in range(25):
        course1, course2, course3 = random.sample(courses, 3)
        
        example = {
            "id": f"course_{i+1}",
            "user_query": f"Which courses can I take if I've completed {course1}?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for courses available after completing {course1}
- This involves prerequisite relationships
- Need to find courses that have {course1} as a prerequisite
- This is a reverse prerequisite lookup

APPROACH STRATEGY:
- Identify the prerequisite relation
- Find all courses that have {course1} as a prerequisite
- Return those course codes
- This is a reverse dependency lookup

CONNECTION TO BYTELOGIC:
- Need prerequisite relation to store prerequisite facts
- Need query to find courses by prerequisite
</thinking>

<pseudocode>
; Problem: Find courses with a specific prerequisite
; Input: Prerequisite course ({course1})
; Goal: Find all courses that have {course1} as a prerequisite

REL prerequisite

; Facts: Course prerequisite relationships
FACT prerequisite {course1} {course2}
FACT prerequisite {course1} {course3}
FACT prerequisite math101 {course2}
FACT prerequisite eng101 cs201

; Rules: None needed for direct lookup

SOLVE
QUERY prerequisite {course1} ?
</pseudocode>

<computation>
REL prerequisite

FACT prerequisite {course1} {course2}
FACT prerequisite {course1} {course3}
FACT prerequisite math101 {course2}
FACT prerequisite eng101 cs201

SOLVE
QUERY prerequisite {course1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_city_connection_examples() -> List[Dict[str, Any]]:
    """Generate city connection examples."""
    examples = []
    
    cities = ["boston", "new_york", "chicago", "washington", "los_angeles", "san_francisco", "seattle", 
              "denver", "miami", "dallas", "houston", "atlanta", "phoenix", "philadelphia", "san_diego"]
    
    for i in range(25):
        city1, city2, city3 = random.sample(cities, 3)
        
        example = {
            "id": f"city_{i+1}",
            "user_query": f"Which cities are connected to {city1} by a direct route?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for cities directly connected to {city1}
- This is a graph connectivity problem
- Looking for direct edges from {city1} to other cities
- This is a direct lookup in a connection graph

APPROACH STRATEGY:
- Identify the connection relation
- Find all facts where {city1} is the source
- Extract destination cities from those facts
- This is a simple adjacency lookup

CONNECTION TO BYTELOGIC:
- Need connected relation to store connection facts
- Need query to extract destinations from source
</thinking>

<pseudocode>
; Problem: Find directly connected cities to a specific city
; Input: Source city ({city1})
; Goal: Find all cities directly connected to {city1}

REL connected

; Facts: City connection relationships
FACT connected {city1} {city2}
FACT connected {city1} {city3}
FACT connected {city2} washington
FACT connected new_york los_angeles

; Rules: None needed for direct lookup

SOLVE
QUERY connected {city1} ?
</pseudocode>

<computation>
REL connected

FACT connected {city1} {city2}
FACT connected {city1} {city3}
FACT connected {city2} washington
FACT connected new_york los_angeles

SOLVE
QUERY connected {city1} ?
</computation>"""
        }
        examples.append(example)
    
    return examples

def generate_product_category_examples() -> List[Dict[str, Any]]:
    """Generate product category examples."""
    examples = []
    
    products = ["laptop", "phone", "tablet", "chair", "desk", "monitor", "keyboard", "mouse", "printer", "scanner"]
    categories = ["electronics", "furniture", "office_supplies", "computers", "mobile", "accessories"]
    
    for i in range(25):
        product1, product2 = random.sample(products, 2)
        category = random.choice(categories)
        
        example = {
            "id": f"product_{i+1}",
            "user_query": f"Which products are in the same category as {product1}?",
            "complete_response": f"""<thinking>
PROBLEM UNDERSTANDING:
- Query asks for products in the same category as '{product1}'
- This involves product categorization
- Need to find the category of {product1}, then other products in that category
- This is a category-based grouping problem

APPROACH STRATEGY:
- Find the category of {product1}
- Find all products in that same category
- Return those products
- This is a category lookup and grouping operation

CONNECTION TO BYTELOGIC:
- Need belongs_to_category relation to store category facts
- Need query to find products by category
</thinking>

<pseudocode>
; Problem: Find products in the same category as a specific product
; Input: Product name ({product1})
; Goal: Find all products in the same category as {product1}

REL belongs_to_category

; Facts: Product-category relationships
FACT belongs_to_category {product1} {category}
FACT belongs_to_category {product2} {category}
FACT belongs_to_category chair furniture
FACT belongs_to_category desk furniture

; Rules: None needed for direct lookup

SOLVE
QUERY belongs_to_category ? {category}
</pseudocode>

<computation>
REL belongs_to_category

FACT belongs_to_category {product1} {category}
FACT belongs_to_category {product2} {category}
FACT belongs_to_category chair furniture
FACT belongs_to_category desk furniture

SOLVE
QUERY belongs_to_category ? {category}
</computation>"""
        }
        examples.append(example)
    
    return examples

def main():
    """Generate the complete dataset."""
    print("Generating large three-tier dataset...")
    
    all_examples = []
    all_examples.extend(generate_family_relationship_examples())
    all_examples.extend(generate_grandparent_examples())
    all_examples.extend(generate_friendship_examples())
    all_examples.extend(generate_workplace_examples())
    all_examples.extend(generate_animal_capability_examples())
    all_examples.extend(generate_course_prerequisite_examples())
    all_examples.extend(generate_city_connection_examples())
    all_examples.extend(generate_product_category_examples())
    
    dataset = {
        "dataset_info": {
            "name": "large_three_tier_bytelogic_dataset",
            "description": "Large dataset with 200 examples for three-tier ByteLogic training",
            "size": len(all_examples),
            "format": "user_query -> <thinking> -> <pseudocode> -> <computation>",
            "phases": ["abstract_planning", "algorithm_design", "implementation_translation", "integrated_training"]
        },
        "examples": all_examples
    }
    
    # Write to file
    with open("data/large-three-tier-dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(all_examples)} examples in total")
    print("Dataset saved to data/large-three-tier-dataset.json")

if __name__ == "__main__":
    main()