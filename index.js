//https://www.w3docs.com/tools/editor/5918
import * as tf from '@tensorflow/tfjs';
var math = require('mathjs');
var $ = require('jquery');
var dt = require('datatables.net')();

import * as raw_anionic_train_Data from "./json/anionic_raw_train_data.json";
const { chem_struct_ANIONIC, Gel_conc_ANIONIC, Gdl_added_ANIONIC } = raw_anionic_train_Data;

import * as raw_cationic_train_Data from "./json/cationic_raw_train_data.json";
const { chem_struct_CATIONIC, Salt_conc_CATIONIC, Gel_conc_CATIONIC, salt_added_CATIONIC } = raw_cationic_train_Data;



function GTP_cationic_api(new_struct_cationic, new_salt_added_cationic, new_salt_concentration_cationic, new_gel_concentration_cationic) {
    const chem_struct_cationic = [...chem_struct_CATIONIC];
    const salt_concentration_cationic = [...Salt_conc_CATIONIC];
    const gel_concentration_cationic = [...Gel_conc_CATIONIC];
    const salt_added_cationic = [...salt_added_CATIONIC];
    var divided_peptides_cationic = []
    var gel_concentration_list_cationic = []
    var salt_concentration_list_cationic = []
    var salt_added_list_cationic = []
    for (let i = 0; i < chem_struct_cationic.length; i++) {
        divided_peptides_cationic.push(str_divide(chem_struct_cationic[i]))
        var gel_concentration_parsed_cationic = gel_concentration_cationic[i].replace(/[^0-9\.]/g, '')
        gel_concentration_list_cationic.push(parseFloat(gel_concentration_parsed_cationic))
        var salt_concentration_parsed_cationic = salt_concentration_cationic[i].replace(/[^0-9\.]/g, '')
        salt_concentration_list_cationic.push(parseFloat(salt_concentration_parsed_cationic))
        salt_added_list_cationic.push(salt_added_cationic[i])
    }
    var ohe_peptides_cationic = ohe_pep_encoder(divided_peptides_cationic)
    var ohe_salt_added_cationic = ohe_salt_encoder(salt_added_list_cationic)
    var normalized_h_dist_cationic = normalized_hamming_dist_finder(ohe_peptides_cationic)
    var normalized_cg_dist_cationic = normalized_conc_dist_finder(gel_concentration_list_cationic) //cg stands for concentration gel
    var normalized_cs_dist_cationic = normalized_conc_dist_finder(salt_concentration_list_cationic) //cs stands for concentration salt
    var normalized_ts_dist_cationic = normalized_hamming_dist_finder(ohe_salt_added_cationic) //ts stands for type salt

    var overall_distance_train_cationic = math.sqrt(math.add(math.dotMultiply(normalized_h_dist_cationic, normalized_h_dist_cationic),
        math.dotMultiply(normalized_cg_dist_cationic, normalized_cg_dist_cationic), math.dotMultiply(normalized_cs_dist_cationic, normalized_cs_dist_cationic), math.dotMultiply(normalized_ts_dist_cationic, normalized_ts_dist_cationic)))
    var overall_distance_train_cationic = tf.tensor2d(overall_distance_train_cationic)
        // Evaluating New test data
    chem_struct_cationic.push(new_struct_cationic)
    salt_added_cationic.push(new_salt_added_cationic)
        //gel_conentration_cationic_new.push(new_gel_conc_cationic)
    var divided_peptides_cationic = []
    var concentration_gel_list_cationic = []
    var concentration_salt_list_cationic = []
    for (let i = 0; i < chem_struct_cationic.length; i++) {
        divided_peptides_cationic.push(str_divide(chem_struct_cationic[i]))
        if (i < chem_struct_cationic.length - 1) {
            var concentration_gel_parsed_cationic = parseFloat(gel_concentration_cationic[i].replace(/[^0-9\.]/g, ''))
            var concentration_salt_parsed_cationic = parseFloat(salt_concentration_cationic[i].replace(/[^0-9\.]/g, ''))
        } else {
            var concentration_gel_parsed_cationic = new_gel_concentration_cationic
            var concentration_salt_parsed_cationic = new_salt_concentration_cationic
        }
        concentration_gel_list_cationic.push(concentration_gel_parsed_cationic)
        concentration_salt_list_cationic.push(concentration_salt_parsed_cationic)
    }
    var ohe_peptides_cationic = ohe_pep_encoder(divided_peptides_cationic)
    var ohe_salt_added_cationic = ohe_salt_encoder(salt_added_cationic)
    var normalized_h_dist_cationic = normalized_hamming_dist_finder(ohe_peptides_cationic)

    var normalized_cg_dist_cationic = normalized_conc_dist_finder(concentration_gel_list_cationic)
    var normalized_cs_dist_cationic = normalized_conc_dist_finder(concentration_salt_list_cationic)
    var normalized_ts_dist_cationic = normalized_hamming_dist_finder(ohe_salt_added_cationic)
    var normalized_h_dist_cationic = normalized_h_dist_cationic[normalized_h_dist_cationic.length - 1].slice(0, normalized_h_dist_cationic.length - 1)
    var normalized_cg_dist_cationic = normalized_cg_dist_cationic[normalized_cg_dist_cationic.length - 1].slice(0, normalized_cg_dist_cationic.length - 1)
    var normalized_cs_dist_cationic = normalized_cs_dist_cationic[normalized_cs_dist_cationic.length - 1].slice(0, normalized_cs_dist_cationic.length - 1)
    var normalized_ts_dist_cationic = normalized_ts_dist_cationic[normalized_ts_dist_cationic.length - 1].slice(0, normalized_ts_dist_cationic.length - 1)
        // console.log('h_dist', normalized_h_dist_cationic)
        // console.log('conc_gel_dist', normalized_cg_dist_cationic)
        // console.log('conc_salt_dist', normalized_cs_dist_cationic)
        // console.log('type_salt_dist', normalized_ts_dist_cationic)
    var overall_distance_test_cationic = math.sqrt(math.add(math.dotMultiply(normalized_h_dist_cationic, normalized_h_dist_cationic),
            math.dotMultiply(normalized_cg_dist_cationic, normalized_cg_dist_cationic), math.dotMultiply(normalized_cs_dist_cationic, normalized_cs_dist_cationic), math.dotMultiply(normalized_ts_dist_cationic, normalized_ts_dist_cationic)))
        //console.log('overall_distance_test', overall_distance_test_cationic)
        // tf.print(overall_distance_train_cationic)
    var overall_distance_test_cationic = tf.tensor1d(overall_distance_test_cationic)
    var overall_distance_test_cationic = tf.reshape(overall_distance_test_cationic, [1, chem_struct_cationic.length - 1])
        //console.log(overall_distance_test_anionic.shape[0])
    var train_cationic_Data = require('./json/cationic_fit_data.json');
    const weights_cationic = tf.tensor2d(train_cationic_Data['weights']);
    // tf.print(overall_distance_train_cationic)
    var result = KRR(overall_distance_test_cationic, overall_distance_train_cationic, weights_cationic)
    console.log(result)
    return result
}
var new_struct_cationic = "Fmoc-Phe-DAP"
var new_gel_concentration_cationic = 20
var new_salt_concentration_cationic = 114
var new_salt_added_cationic = "NaCl"

var result = GTP_cationic_api(new_struct_cationic, new_salt_added_cationic, new_salt_concentration_cationic, new_gel_concentration_cationic)

var new_struct_cationic = "Fmoc-Phe-OH"
var new_gel_concentration_cationic = 7.5
var new_gdl_added = 2
var result = GTP_anionic_api(new_struct_cationic, new_gel_concentration_cationic, new_gdl_added)

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


function KRR(overall_distance_test, overall_distance_train, weights) {
    //console.log(overall_distance_test.shape[0])
    console.log('overall_distance_train')
    tf.print(overall_distance_train.shape)
    console.log('overall_distance_test')
    tf.print(overall_distance_test.shape)
    console.log('weights')
    tf.print(weights.shape)
    var O_distance_test = overall_distance_test
    var O_distance_train = overall_distance_train
    var length_test_data = O_distance_test.shape[0]
    var W = weights
    var i;
    for (i = 0; i < length_test_data; i++) {
        var K = tf.matMul(
            tf.gather(O_distance_test, [i]),
            O_distance_train
        )
        tf.print(K)
        var prediction = tf.matMul(K, W)
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

function ohe_salt_encoder(salt_added_list_cationic) { //input is salt type data
    // Fidning all possible classes
    var Salt_type_classes = []
    for (let i = 0; i < salt_added_list_cationic.length; i++) {
        Salt_type_classes.push(salt_added_list_cationic[i])
    }
    var Salt_type_classes = Salt_type_classes.filter(onlyUnique).sort();
    /// Finding one-hot encoding
    var ohe_salts_added = []
    for (let k = 0; k < salt_added_list_cationic.length; k++) {
        var ohe_salt = Array(Salt_type_classes.length).fill(0)

        for (let j = 0; j < Salt_type_classes.length; j++) {
            //var check = possible_classes[i][j].includes(divided_peptides[0][i]);
            if (Salt_type_classes[j] == salt_added_list_cationic[k]) {
                ohe_salt[j] = 1
            }
        }
        ohe_salts_added.push(ohe_salt)
    }
    return ohe_salts_added
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
    const chem_struct_anionic = [...chem_struct_ANIONIC];
    const concentration_anionic = [...Gel_conc_ANIONIC];
    const Gdl_anionic = [...Gdl_added_ANIONIC];


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
    const weights_anionic = tf.tensor2d(train_anionic_Data['weights']);
    var result = KRR(overall_distance_test_anionic, overall_distance_train_anionic, weights_anionic)
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


$("#new_gel_type").change(function() {

    $('#prediction_box').hide();
    $('#notes').hide();
    var NEW_gel_type = document.getElementById("new_gel_type").value
    if (NEW_gel_type == "anionic") {
        $('#chem_struc_anionic_field').show();
        $('#chem_struc_cationic_field').hide();
        $('#gel_conc_anionic_field').show();
        $('#gel_conc_cationic_field').hide();
        $('#equivalent_gdl_field').show();
        $('#salt_type_field').hide();
        $('#salt_conc_field').hide();

        document.getElementById("new_salt_type").required = false;
        document.getElementById("new_salt_added_cationic").required = false;
        document.getElementById("new_gel_conc_anionic").required = true;
        document.getElementById("new_gel_conc_cationic").required = false;
        document.getElementById("new_Gdl_added_anionic").required = true;
        document.getElementById("new_chem_struct_anionic").required = true;
        document.getElementById("new_chem_struct_cationic").required = false;
        $("#input").css("height", "380px");


    } else if (NEW_gel_type == "cationic") {
        $('#chem_struc_anionic_field').hide();
        $('#chem_struc_cationic_field').show();
        $('#gel_conc_anionic_field').hide();
        $('#gel_conc_cationic_field').show();
        $('#salt_type_field').show();
        $('#salt_conc_field').show();
        $('#equivalent_gdl_field').hide();
        document.getElementById("new_gel_conc_anionic").required = false;
        document.getElementById("new_gel_conc_cationic").required = true;
        document.getElementById("new_Gdl_added_anionic").required = true;
        document.getElementById("new_salt_type").required = true;
        document.getElementById("new_salt_added_cationic").required = true;
        document.getElementById("new_Gdl_added_anionic").required = false;
        document.getElementById("new_chem_struct_anionic").required = false;
        document.getElementById("new_chem_struct_cationic").required = true;
        $("#input").css("height", "450px");


    } else {
        document.getElementById("chem_struc_anionic_field").style.display = "none"
        document.getElementById("chem_struc_cationic_field").style.display = "none"
        document.getElementById("gel_conc_anionic_field").style.display = "none"
        document.getElementById("gel_conc_cationic_field").style.display = "none"
        document.getElementById("equivalent_gdl_field").style.display = "none"
        document.getElementById("salt_type_field").style.display = "none"
        document.getElementById("salt_conc_field").style.display = "none"

    }
});
$("#new_gel_type").trigger("change");


document.addEventListener("DOMContentLoaded", function() {
    var form = document.getElementById("input");
    var output = document.getElementById("output");
    var button = document.getElementById("getDataBtn");
    var new_gel_type = document.getElementById("new_gel_type").value
    console.log(new_gel_type)
        //var prediction_score = document.getElementById("prediction_score");
        //var prediction_trancparency = document.getElementById("prediction_transparency");
    form.addEventListener("submit", function(e) {
        e.preventDefault();
        //e.stopPropagation();
        var new_gel_type = document.getElementById("new_gel_type").value

        if (new_gel_type == 'anionic') {
            // document.getElementById("salt_type_field").required = false;
            // $("#salt_conc_field").removeAttr('required');
            var new_struct_anionic = document.getElementById("new_chem_struct_anionic").value;
            var new_concentration_anionic = document.getElementById("new_gel_conc_anionic").value;
            var new_Gdl_anionic = document.getElementById("new_Gdl_added_anionic").value;
            var result = GTP_anionic_api(new_struct_anionic, new_concentration_anionic, new_Gdl_anionic);
            //prediction_score.innerHTML = result[1];
            //prediction_trancparency.innerHTML = result[0];
            document.getElementById("report_avg_score").innerHTML = "Average score for the anionic model is currently 0.83.";
            document.getElementById("message").innerHTML = "Score shows the certainty of the model for new predictions on the scale of 0 to 1.";
            document.getElementById("prediction_transparency").innerHTML = "Prediction: " + "&nbsp" + "&nbsp" + "&nbsp" + result[0]
            document.getElementById("prediction_score").innerHTML = "Score: " + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + result[1]
            var el = document.getElementById('prediction_box');
            //el.style.backgroundColor = "#ff0000";
            el.style.cssText = 'position: absolute; left: 350px; width:300px; top: 160px; background-color: #fffccc; padding:5px 10px 5px 20px; border: 3px solid #ffb99c; border-radius: 10px;'
                //document.getElementById("prediction_box").style.display = "none"
            $('#prediction_box').show();
            $('#notes').show();
        }
        if (new_gel_type == 'cationic') {

            // $("#salt_conc_field").removeAttr('required');
            var new_struct_cationic = document.getElementById("new_chem_struct_anionic").value;
            var new_salt_added_cationic = document.getElementById("new_salt_type").value;
            var new_salt_concentration_cationic = document.getElementById("new_salt_added_cationic").value;
            var new_gel_concentration_cationic = document.getElementById("new_gel_conc_cationic").value;
            var result = GTP_cationic_api(new_struct_cationic, new_salt_added_cationic, new_salt_concentration_cationic, new_gel_concentration_cationic)
                //prediction_score.innerHTML = result[1];
                //prediction_trancparency.innerHTML = result[0];
            document.getElementById("report_avg_score").innerHTML = "Average score for the cationic model is currently 0.86.";
            document.getElementById("message").innerHTML = "Score shows the certainty of the model for new predictions on the scale of 0 to 1.";
            document.getElementById("prediction_transparency").innerHTML = "Prediction: " + "&nbsp" + "&nbsp" + "&nbsp" + result[0]
            document.getElementById("prediction_score").innerHTML = "Score: " + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + "&nbsp" + result[1]
            var el = document.getElementById('prediction_box');
            //el.style.backgroundColor = "#ff0000";
            el.style.cssText = 'position: absolute; left: 350px; width:300px; top: 160px; background-color: #fffccc; padding:5px 10px 5px 20px; border: 3px solid #ffb99c; border-radius: 10px;'
                //document.getElementById("prediction_box").style.display = "none"
            $('#prediction_box').show();
            $('#notes').show();
        }
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