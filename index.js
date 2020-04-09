//https://www.w3docs.com/tools/editor/5918
import * as raw_anionic_train_Data from "./json/anionic_raw_train_data.json";
const { chem_struct, Gel_conc, Gdl_added } = raw_anionic_train_Data;

import * as tf from '@tensorflow/tfjs';
var math = require('mathjs');
var $ = require('jquery');
var dt = require('datatables.net')();

function kernel_linear(a, b) {
    const dot_product = tf.mul(a, b);
    return dot_product; // The function returns the dot product of a and b
}

function compute_kernel(a, b) {
    // Sample size
    const n_a = a.shape[0];
    const n_b = b.shape[0];

    // Finding the gram matrix
    var K = tf.zeros([n_a, n_b]);
    var i,
        j;
    for (i = 0; i < n_a; i++) {
        for (j = 0; j < n_b; j++) {
            // K[i, j] = kernel_linear(a.gather(i), b.gather(j)) tf.tensor1d(a[i]).print()
            // console.log(i); K.gather(i,j)
        }
    }
    return K;
}

// var K = tf.zeros([3, 3]); tf.gatherND(K, [2, 2]).print(); tf.print(d)
// function Loading data from json file;



function GTP_anionic(overall_distance_test, overall_distance_train, weights) {
    //console.log(overall_distance_test.shape[0])
    var length_test_data = overall_distance_test.shape[0]
    var i;
    for (i = 0; i < length_test_data; i++) {
        var K = tf.matMul(
            tf.gather(overall_distance_test, [i]),
            overall_distance_train
        )
        var prediction = tf.matMul(K, weights)
            //console.log(prediction.dataSync())
        var max_index = prediction
            .argMax(1)
            .dataSync()

        var index_m = 1
            //tf.gatherND(prediction, [0, 1]) tf.sum(tf.abs(prediction))

        var score = tf.div(
            tf.gatherND(prediction, [0, max_index]),
            tf.sum(tf.abs(prediction))
        )
        var a = tf.zeros([1, 2])
        a = tf.where(tf.equal(prediction.max(1), prediction), tf.onesLike(a), a)
            //var  tf.gatherND(prediction, [0, 0]) console.log(max_index[0])
        if (max_index[0] == 0) {
            var labeled_prediction = 'Non-Transparent'
        } else {
            var labeled_prediction = 'Transparent'
        }
        console.log(
            'label:' + labeled_prediction + '.',
            'Score on prediction is: ' + Math.round(score.dataSync() * 1000) / 1000
        )
    }
    return [labeled_prediction, Math.round(score.dataSync() * 1000) / 1000]
}

function str_divide(chem_struct) {
    var modified_str = chem_struct.replace(/-/gi, '').split('Phe', 3);
    var parsed_modified_str = [modified_str[0].slice(0, 4), modified_str[0].slice(4, modified_str[0].length), modified_str[1]];
    //first item: N Terminus, Second item: functional group, Last item: C terminus
    return parsed_modified_str;
}

function ohe_pep_encoder(divided_peptides) { //input is parsed peptide data
    // Fidning all possible classes
    var N_terminus_classes = []
    var Func_classes = []
    var C_terminus_classes = []
    for (let i = 0; i < divided_peptides.length; i++) {
        for (let j = 0; j < divided_peptides[j].length; j++) {
            if (j == 0) { N_terminus_classes.push(divided_peptides[i][j]); }
            if (j == 1) { Func_classes.push(divided_peptides[i][j]); }
            if (j == 2) { C_terminus_classes.push(divided_peptides[i][j]); }
        }
    }
    var N_terminus_classes = N_terminus_classes.filter(onlyUnique).sort();
    var Func_classes = Func_classes.filter(onlyUnique).sort();
    var C_terminus_classes = C_terminus_classes.filter(onlyUnique).sort();
    var n_classes = N_terminus_classes.length + Func_classes.length + C_terminus_classes.length
    var possible_classes = [N_terminus_classes, Func_classes, C_terminus_classes]

    /// Finding one-hot encoding
    var ohe_peptides = []
    for (let k = 0; k < divided_peptides.length; k++) {
        var ohe_peptide = Array(n_classes).fill(0)
        for (let i = 0; i < divided_peptides[k].length; i++) {
            for (let j = 0; j < possible_classes[i].length; j++) {
                //var check = possible_classes[i][j].includes(divided_peptides[0][i]);
                if (possible_classes[i][j] == divided_peptides[k][i]) {
                    if (i == 0) { ohe_peptide[j] = 1 }
                    if (i == 1) { ohe_peptide[j + N_terminus_classes.length] = 1 }
                    if (i == 2) { ohe_peptide[j + N_terminus_classes.length + Func_classes.length] = 1 }
                }

            }
        }
        ohe_peptides.push(ohe_peptide)
    }
    return ohe_peptides
}

function hamming_distance(a, b, Length) { //Length here is the number of possible_classes.length
    var c = math.dotMultiply(a, b)
    var d = Length - math.sum(c)
    return d;
}

function normalized_hamming_dist_finder(ohe_peptides) {
    var Length = math.sum(ohe_peptides[0])
    var normalized_h_dist = []
    var max_h_dist = []
    for (let i = 0; i < ohe_peptides.length; i++) {
        var h_dist_list = []
        for (let j = 0; j < ohe_peptides.length; j++) {
            var h_dist = hamming_distance(ohe_peptides[i], ohe_peptides[j], Length)
            h_dist_list.push(h_dist)
        }
        max_h_dist.push(Math.max(...h_dist_list))
            //var h_dist_list_max = Math.max(...h_dist_list[i])
        normalized_h_dist.push(math.divide(h_dist_list, max_h_dist[i]))
    }
    return normalized_h_dist;
}

function normalized_conc_dist_finder(concentration_list) {
    var normalized_c_dist = []
    var max_conc = Math.max(...concentration_list)
    for (let i = 0; i < concentration_list.length; i++) {
        var c_dist_list = []
        for (let j = 0; j < concentration_list.length; j++) {
            var conc_dist = concentration_list[i] - concentration_list[j]
            c_dist_list.push(conc_dist)
        }
        //var h_dist_list_max = Math.max(...h_dist_list[i])
        normalized_c_dist.push(math.divide(c_dist_list, max_conc))
    }
    return normalized_c_dist;
}

function onlyUnique(value, index, self) {
    return self.indexOf(value) === index;
}

function GTP_anionic_api(new_struct_anionic, new_concentration_anionic, new_Gdl_anionic) {
    //Preparing inputs
    // var raw_anionic_train_Data = require('./json/anionic_raw_train_data.json');
    // var chem_struct_anionic = raw_anionic_train_Data['chem_struct']
    // var concentration_anionic = raw_anionic_train_Data['Gel_conc'];
    // var Gdl_anionic = raw_anionic_train_Data['Gdl_added'];
    const chem_struct_anionic = [...chem_struct];
    const concentration_anionic = [...Gel_conc];
    const Gdl_anionic = [...Gdl_added];


    var divided_peptides_anionic = []
    var concentration_list_anionic = []
    for (let i = 0; i < chem_struct_anionic.length; i++) {
        divided_peptides_anionic.push(str_divide(chem_struct_anionic[i]))
        var concentration_parsed_anionic = concentration_anionic[i].replace(/[^0-9\.]/g, '')
        concentration_list_anionic.push(parseFloat(concentration_parsed_anionic))
    }

    var ohe_peptides_anionic = ohe_pep_encoder(divided_peptides_anionic)
    var normalized_h_dist_anionic = normalized_hamming_dist_finder(ohe_peptides_anionic)
    var normalized_c_dist_anionic = normalized_conc_dist_finder(concentration_list_anionic)
    var normalized_Gdl_dist_anionic = normalized_conc_dist_finder(Gdl_anionic)

    var overall_distance_train_anionic = math.sqrt(math.add(math.dotMultiply(normalized_h_dist_anionic, normalized_h_dist_anionic),
        math.dotMultiply(normalized_c_dist_anionic, normalized_c_dist_anionic), math.dotMultiply(normalized_Gdl_dist_anionic, normalized_Gdl_dist_anionic)))
    var overall_distance_train_anionic = tf.tensor2d(overall_distance_train_anionic)

    // Evaluating New test data
    //var new_struct_anionic = 'Fmoc-Phe-OH'
    //var new_concentration_anionic = 7.5
    //var new_Gdl_anionic = 2
    chem_struct_anionic.push(new_struct_anionic)
        //concentration_anionic.push(new_concentration_anionic)

    Gdl_anionic.push(new_Gdl_anionic)
    var divided_peptides_anionic = []
    var concentration_list_anionic = []
    for (let i = 0; i < chem_struct_anionic.length; i++) {
        divided_peptides_anionic.push(str_divide(chem_struct_anionic[i]))
        if (i < chem_struct_anionic.length - 1) {
            var concentration_parsed_anionic = parseFloat(concentration_anionic[i].replace(/[^0-9\.]/g, ''))
        } else { concentration_parsed_anionic = new_concentration_anionic }
        concentration_list_anionic.push(concentration_parsed_anionic)
    }
    var ohe_peptides_anionic = ohe_pep_encoder(divided_peptides_anionic)
    var normalized_h_dist_anionic = normalized_hamming_dist_finder(ohe_peptides_anionic)
    var normalized_c_dist_anionic = normalized_conc_dist_finder(concentration_list_anionic)

    var normalized_Gdl_dist_anionic = normalized_conc_dist_finder(Gdl_anionic)
        // considering the distance of the added data to all the trained data

    //console.log(normalized_c_dist_anionic[normalized_c_dist_anionic.length - 1].slice(0, normalized_c_dist_anionic.length - 1))
    var normalized_h_dist_anionic = normalized_h_dist_anionic[normalized_h_dist_anionic.length - 1].slice(0, normalized_h_dist_anionic.length - 1)
    var normalized_c_dist_anionic = normalized_c_dist_anionic[normalized_c_dist_anionic.length - 1].slice(0, normalized_c_dist_anionic.length - 1)
    var normalized_Gdl_dist_anionic = normalized_Gdl_dist_anionic[normalized_Gdl_dist_anionic.length - 1].slice(0, normalized_Gdl_dist_anionic.length - 1)

    var overall_distance_test_anionic = math.sqrt(math.add(math.dotMultiply(normalized_h_dist_anionic, normalized_h_dist_anionic),
        math.dotMultiply(normalized_c_dist_anionic, normalized_c_dist_anionic), math.dotMultiply(normalized_Gdl_dist_anionic, normalized_Gdl_dist_anionic)))


    var overall_distance_test_anionic = tf.tensor1d(overall_distance_test_anionic)

    var overall_distance_test_anionic = tf.reshape(overall_distance_test_anionic, [1, Gdl_anionic.length - 1])
        //console.log(overall_distance_test_anionic.shape[0])
    var train_anionic_Data = require('./json/anionic_fit_data.json');
    const weights = tf.tensor2d(train_anionic_Data['weights']);
    var result = GTP_anionic(overall_distance_test_anionic, overall_distance_train_anionic, weights)
    return (result)
}


// var new_struct_anionic = 'Fmoc-4-Me-Phe-OH'
// var new_concentration_anionic = 15
// var new_Gdl_anionic = 1
// var result = GTP_anionic_api(new_struct_anionic, new_concentration_anionic, new_Gdl_anionic)




//var test_anionic_Data = require('./json/anionic_test_data.json');
// var overall_distance_train = tf.tensor2d(
//     train_anionic_Data['overall_distance_train']
// );
// var overall_distance_test = tf.tensor2d(
//     test_anionic_Data['overall_distance_test']
// );




function toJSONString(form) {
    var obj = {};
    var elements = form.querySelectorAll("input");
    for (var i = 0; i < elements.length; ++i) {
        var element = elements[i];
        var name = element.name;
        var value = element.value;

        if (name) {
            obj[name] = value;
        }
    }

    return JSON.stringify(obj);
}


document.addEventListener("DOMContentLoaded", function() {
    var form = document.getElementById("input");
    var output = document.getElementById("output");
    var button = document.getElementById("getDataBtn");
    //var prediction_score = document.getElementById("prediction_score");
    //var prediction_trancparency = document.getElementById("prediction_transparency");
    form.addEventListener("submit", function(e) {
        e.preventDefault();
        //e.stopPropagation();
        var new_struct_anionic = document.getElementById("new_chem_struct_anionic").value;
        var new_concentration_anionic = document.getElementById("new_gel_conc_anionic").value;
        var new_Gdl_anionic = document.getElementById("new_Gdl_added_anionic").value;
        var result = GTP_anionic_api(new_struct_anionic, new_concentration_anionic, new_Gdl_anionic);
        //prediction_score.innerHTML = result[1];
        //prediction_trancparency.innerHTML = result[0];
        document.getElementById("message").innerHTML = "Score shows the certainty of the model on new predictions on the scale of 0 to 1.";
        document.getElementById("prediction_transparency").innerHTML = "Prediction: " + "&nbsp" + "&nbsp" + "&nbsp" + result[0]
        document.getElementById("prediction_score").innerHTML = "Score: " + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + result[1]
        var el = document.getElementById('prediction_box');
        //el.style.backgroundColor = "#ff0000";
        el.style.cssText = 'position: absolute; left: 300px; top: 145px; background-color: #fffccc; padding:5px 15px 5px 15px; border: 3px solid #ffb99c; border-radius: 10px;'
    }, false);

});

// document.addEventListener("DOMContentLoaded", function() {
//     var form = document.getElementById("input");
//     var output = document.getElementById("output");
//     form.addEventListener("submit", function(e) {
//         e.preventDefault();
//         e.stopPropagation();
//         var new_struct_anionic = document.getElementById("new_chem_struct_anionic").value;
//         var new_concentration_anionic = document.getElementById("new_gel_conc_anionic").value;
//         var new_Gdl_anionic = document.getElementById("new_Gdl_added_anionic").value;
//         var result = GTP_anionic_api(new_struct_anionic, new_concentration_anionic, new_Gdl_anionic)

//         var json = toJSONString(this);
//         output.innerHTML = result;
//         document.getElementById('input').reset()
//     }, false);

// });