/**
 * BSL Sign Definitions for Recording Tool
 * This mirrors the Python definitions in /src/sign_definitions.py
 */

const CATEGORIES = [
    { id: 'alphabet', name: 'Alphabet', icon: '🔤' },
    { id: 'numbers', name: 'Numbers', icon: '🔢' },
    { id: 'greetings', name: 'Greetings', icon: '👋' },
    { id: 'people', name: 'Family', icon: '👨‍👩‍👧‍👦' },
    { id: 'food_drink', name: 'Food & Drink', icon: '🍽️' },
    { id: 'colors', name: 'Colours', icon: '🎨' },
    { id: 'time', name: 'Time', icon: '⏰' },
    { id: 'emotions', name: 'Feelings', icon: '😊' },
];

const SIGN_DEFINITIONS = [
    // Alphabet (A-Z)
    { id: 'bsl_alphabet_a', name: 'A', category: 'alphabet', twoHanded: true, description: 'Point index finger at open palm' },
    { id: 'bsl_alphabet_b', name: 'B', category: 'alphabet', twoHanded: true, description: 'Point at extended fingers of flat hand' },
    { id: 'bsl_alphabet_c', name: 'C', category: 'alphabet', twoHanded: true, description: 'Curved hand forming C shape' },
    { id: 'bsl_alphabet_d', name: 'D', category: 'alphabet', twoHanded: true, description: 'Index touches curved index finger' },
    { id: 'bsl_alphabet_e', name: 'E', category: 'alphabet', twoHanded: true, description: 'Index touches bent fingers' },
    { id: 'bsl_alphabet_f', name: 'F', category: 'alphabet', twoHanded: true, description: 'Two fingers on index' },
    { id: 'bsl_alphabet_g', name: 'G', category: 'alphabet', twoHanded: true, description: 'Point at fist with extended index' },
    { id: 'bsl_alphabet_h', name: 'H', category: 'alphabet', twoHanded: true, description: 'Two fingers across palm' },
    { id: 'bsl_alphabet_i', name: 'I', category: 'alphabet', twoHanded: true, description: 'Little finger on palm' },
    { id: 'bsl_alphabet_j', name: 'J', category: 'alphabet', twoHanded: true, description: 'Little finger traces J shape' },
    { id: 'bsl_alphabet_k', name: 'K', category: 'alphabet', twoHanded: true, description: 'V-shape on index finger' },
    { id: 'bsl_alphabet_l', name: 'L', category: 'alphabet', twoHanded: true, description: 'L-shape with thumb on palm' },
    { id: 'bsl_alphabet_m', name: 'M', category: 'alphabet', twoHanded: true, description: 'Three fingers on palm' },
    { id: 'bsl_alphabet_n', name: 'N', category: 'alphabet', twoHanded: true, description: 'Two fingers on palm' },
    { id: 'bsl_alphabet_o', name: 'O', category: 'alphabet', twoHanded: true, description: 'O shape touching hand' },
    { id: 'bsl_alphabet_p', name: 'P', category: 'alphabet', twoHanded: true, description: 'Fingers point down between fingers' },
    { id: 'bsl_alphabet_q', name: 'Q', category: 'alphabet', twoHanded: true, description: 'Bent index at palm base' },
    { id: 'bsl_alphabet_r', name: 'R', category: 'alphabet', twoHanded: true, description: 'Crossed fingers on palm' },
    { id: 'bsl_alphabet_s', name: 'S', category: 'alphabet', twoHanded: true, description: 'Fist with thumb across' },
    { id: 'bsl_alphabet_t', name: 'T', category: 'alphabet', twoHanded: true, description: 'Index under thumb on palm' },
    { id: 'bsl_alphabet_u', name: 'U', category: 'alphabet', twoHanded: true, description: 'Two fingers together at palm' },
    { id: 'bsl_alphabet_v', name: 'V', category: 'alphabet', twoHanded: true, description: 'V-shape spread on palm' },
    { id: 'bsl_alphabet_w', name: 'W', category: 'alphabet', twoHanded: true, description: 'Three fingers spread on palm' },
    { id: 'bsl_alphabet_x', name: 'X', category: 'alphabet', twoHanded: true, description: 'Hooked index on palm' },
    { id: 'bsl_alphabet_y', name: 'Y', category: 'alphabet', twoHanded: true, description: 'Thumb and pinky extended' },
    { id: 'bsl_alphabet_z', name: 'Z', category: 'alphabet', twoHanded: true, description: 'Index traces Z shape' },

    // Numbers (0-10)
    { id: 'bsl_numbers_0', name: '0', category: 'numbers', twoHanded: false, description: 'Closed fist facing viewer' },
    { id: 'bsl_numbers_1', name: '1', category: 'numbers', twoHanded: false, description: 'Index finger extended' },
    { id: 'bsl_numbers_2', name: '2', category: 'numbers', twoHanded: false, description: 'Index and middle in V-shape' },
    { id: 'bsl_numbers_3', name: '3', category: 'numbers', twoHanded: false, description: 'Three fingers extended' },
    { id: 'bsl_numbers_4', name: '4', category: 'numbers', twoHanded: false, description: 'Four fingers extended' },
    { id: 'bsl_numbers_5', name: '5', category: 'numbers', twoHanded: false, description: 'All five fingers extended' },
    { id: 'bsl_numbers_6', name: '6', category: 'numbers', twoHanded: true, description: 'Five plus one finger' },
    { id: 'bsl_numbers_7', name: '7', category: 'numbers', twoHanded: true, description: 'Five plus two fingers' },
    { id: 'bsl_numbers_8', name: '8', category: 'numbers', twoHanded: true, description: 'Five plus three fingers' },
    { id: 'bsl_numbers_9', name: '9', category: 'numbers', twoHanded: true, description: 'Five plus four fingers' },
    { id: 'bsl_numbers_10', name: '10', category: 'numbers', twoHanded: true, description: 'Both hands showing five' },

    // Greetings
    { id: 'bsl_greetings_hello', name: 'Hello', category: 'greetings', twoHanded: false, description: 'Wave with palm outward' },
    { id: 'bsl_greetings_goodbye', name: 'Goodbye', category: 'greetings', twoHanded: false, description: 'Wave side to side' },
    { id: 'bsl_greetings_thank_you', name: 'Thank You', category: 'greetings', twoHanded: false, description: 'Hand from chin forward' },
    { id: 'bsl_greetings_please', name: 'Please', category: 'greetings', twoHanded: false, description: 'Hand circles on chest' },
    { id: 'bsl_greetings_sorry', name: 'Sorry', category: 'greetings', twoHanded: false, description: 'Fist circles on chest' },
    { id: 'bsl_greetings_yes', name: 'Yes', category: 'greetings', twoHanded: false, description: 'Fist nods up and down' },
    { id: 'bsl_greetings_no', name: 'No', category: 'greetings', twoHanded: false, description: 'Fingers close onto thumb' },

    // Family/People
    { id: 'bsl_people_mother', name: 'Mother', category: 'people', twoHanded: false, description: 'Tap side of chin twice' },
    { id: 'bsl_people_father', name: 'Father', category: 'people', twoHanded: false, description: 'Tap forehead twice' },
    { id: 'bsl_people_brother', name: 'Brother', category: 'people', twoHanded: true, description: 'Two fists moving apart' },
    { id: 'bsl_people_sister', name: 'Sister', category: 'people', twoHanded: true, description: 'Crossed index fingers apart' },
    { id: 'bsl_people_baby', name: 'Baby', category: 'people', twoHanded: true, description: 'Cradling motion' },
    { id: 'bsl_people_family', name: 'Family', category: 'people', twoHanded: true, description: 'F-shapes circle around' },

    // Food & Drink
    { id: 'bsl_food_eat', name: 'Eat', category: 'food_drink', twoHanded: false, description: 'Bunched fingers to mouth' },
    { id: 'bsl_food_drink', name: 'Drink', category: 'food_drink', twoHanded: false, description: 'C-shape tips to mouth' },
    { id: 'bsl_food_water', name: 'Water', category: 'food_drink', twoHanded: false, description: 'W-shape taps chin' },
    { id: 'bsl_food_tea', name: 'Tea', category: 'food_drink', twoHanded: true, description: 'Dipping tea bag motion' },
    { id: 'bsl_food_coffee', name: 'Coffee', category: 'food_drink', twoHanded: true, description: 'Grinding coffee motion' },
    { id: 'bsl_food_hungry', name: 'Hungry', category: 'food_drink', twoHanded: false, description: 'Hand down chest' },

    // Colours
    { id: 'bsl_colors_red', name: 'Red', category: 'colors', twoHanded: false, description: 'Index strokes from lip' },
    { id: 'bsl_colors_blue', name: 'Blue', category: 'colors', twoHanded: false, description: 'B-shape twists at wrist' },
    { id: 'bsl_colors_green', name: 'Green', category: 'colors', twoHanded: false, description: 'G-shape twists at wrist' },
    { id: 'bsl_colors_yellow', name: 'Yellow', category: 'colors', twoHanded: false, description: 'Y-shape shakes side to side' },
    { id: 'bsl_colors_black', name: 'Black', category: 'colors', twoHanded: false, description: 'Index across eyebrow' },
    { id: 'bsl_colors_white', name: 'White', category: 'colors', twoHanded: false, description: 'Hand brushes down chest' },

    // Time
    { id: 'bsl_time_today', name: 'Today', category: 'time', twoHanded: true, description: 'Both hands move down' },
    { id: 'bsl_time_tomorrow', name: 'Tomorrow', category: 'time', twoHanded: false, description: 'Thumb on cheek arcs forward' },
    { id: 'bsl_time_yesterday', name: 'Yesterday', category: 'time', twoHanded: false, description: 'Thumb moves backward' },
    { id: 'bsl_time_morning', name: 'Morning', category: 'time', twoHanded: true, description: 'Hand rises from elbow' },
    { id: 'bsl_time_night', name: 'Night', category: 'time', twoHanded: true, description: 'Hand descends past elbow' },
    { id: 'bsl_time_week', name: 'Week', category: 'time', twoHanded: true, description: 'Index slides along palm' },

    // Feelings/Emotions
    { id: 'bsl_emotions_happy', name: 'Happy', category: 'emotions', twoHanded: false, description: 'Hand circles on chest' },
    { id: 'bsl_emotions_sad', name: 'Sad', category: 'emotions', twoHanded: false, description: 'Hand moves down face' },
    { id: 'bsl_emotions_tired', name: 'Tired', category: 'emotions', twoHanded: true, description: 'Hands on chest drop down' },
    { id: 'bsl_emotions_good', name: 'Good', category: 'emotions', twoHanded: false, description: 'Thumbs up or hand from chin' },
    { id: 'bsl_emotions_bad', name: 'Bad', category: 'emotions', twoHanded: false, description: 'Hand away from chin' },
    { id: 'bsl_emotions_help', name: 'Help', category: 'emotions', twoHanded: true, description: 'Fist on palm, both rise' },
];

// Utility functions
function getSignsByCategory(category) {
    if (!category) return SIGN_DEFINITIONS;
    return SIGN_DEFINITIONS.filter(sign => sign.category === category);
}

function getSignById(signId) {
    return SIGN_DEFINITIONS.find(sign => sign.id === signId);
}

function getCategoryById(categoryId) {
    return CATEGORIES.find(cat => cat.id === categoryId);
}
