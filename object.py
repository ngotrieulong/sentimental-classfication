import re
people_list = []
place_list = []
facility_list = []
service_list = []
general_list = []
specials = r"[->\[\]\'\\\n\.\.\.\?\"\“\,\,>>():;!@#$%^&*?~/`<>{}=+-]"

with open('label_people.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        i = re.sub(specials, '', i)
        people_list.append(i)

with open('label_place.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        i = re.sub(specials, '', i)
        place_list.append(i)

with open('label_facility.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        i = re.sub(specials, '', i)
        facility_list.append(i)

with open('label_service.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        i = re.sub(specials, '', i)
        service_list.append(i)

with open('label_general.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        i = re.sub(specials, '', i)
        general_list.append(i)

people_total = 0
place_total = 0
facility_total = 0
service_total = 0
general_total = 0
with open('daura.csv', 'r', encoding='utf8') as f:
    for row in f:
        print('====================')
        print(row)
        people_count = 0
        place_count = 0
        facility_count = 0
        service_count = 0
        people_check = True
        place_check = True
        facility_check = True
        service_check = True
        general_check = True

        for word in row.split():
            if people_check:
                if people_list.count(word) >= 1:
                    people_check = False
                    people_total += 1
                    print('--People')
            if place_check:
                if place_list.count(word) >= 1:
                    place_check = False
                    place_total += 1
                    print('--Place')
            if facility_check:
                if facility_list.count(word) >= 1:
                    facility_check = False
                    facility_total += 1
                    print('--Facility')
            if service_check:
                if service_list.count(word) >= 1:
                    service_check = False
                    service_total += 1
                    print('--Service')
            if general_check:
                if general_list.count(word) >= 1:
                    general_check = False
                    general_total += 1
                    print('--General')
    print('\nTOTAL:\nPeople: ' + str(people_total) + '\nPlace: ' + str(place_total))
    print('Facility: ' + str(facility_total) + '\nService: ' + str(service_total) + '\nGeneral: ' + str(general_total))
