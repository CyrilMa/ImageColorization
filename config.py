dataset_conf = {
    "SUN2012" : {
        "n_classes":52,
        "path":"data/SUN2012/Images/",
        "labels":['youth_hostel', 'office', 'ocean', 'bar', 'building_facade', 
        'bathroom', 'basement', 'beauty_salon', 'beach', 'bedroom', 'mountain_snowy',
         'mountain', 'creek', 'classroom', 'closet', 'corridor', 'childs_room', 
         'clothing_store', 'conference_room', 'coast', 'wet_bar', 'window_seat', 
         'waiting_room', 'dorm_room', 'dining_room', 'hotel_room', 'home_office', 
         'highway', 'auditorium', 'alley', 'art_gallery', 'abbey', 'artists_loft', 
         'art_studio', 'attic', 'airport_terminal', 'pasture', 'parlor', 'playroom', 
         'galley', 'game_room', 'utility_room', 'river', 'reception', 'kitchen', 'living_room', 
        'valley', 'skyscraper', 'staircase', 'street', 'shoe_shop', 'nursery'],
    },

    "short_SUN2012": {
        "n_classes":13,
        "path":"data/SUN2012/Images/",
        "labels":['office', 'ocean', 'building_facade', 'beach', 'mountain_snowy',
         'mountain', 'creek',  'highway', 'pasture', 'river', 'valley', 'skyscraper',
          'street'],
    },

    "MIT-places" : {
        "n_classes":4,
        "path":"data/scenedatabase",
        "labels":['Opencountry', 'coast', 'forest', 'mountain'],
    },
}