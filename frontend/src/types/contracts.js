/** @typedef {{id:number,code:string,amount:number,distance:number,holiday:boolean,label:0|1|null,note:string}} Claim */
/** @typedef {{id:number,name:string,threshold:number,state:string,result:object}} Training */
/** @typedef {{id:number,claim_id:number,training_id:number,score:number,decision:string,status:string}} Review */
/**
 * @typedef {Object} Opinion
 * @property {number} id
 * @property {number} claim_id
 * @property {string} annotator
 * @property {0|1} label
 * @property {string} note
 * @property {number} revision
 */
/**
 * @typedef {Object} CalibrationClaim
 * @property {number} claim_id
 * @property {string} claim_code
 * @property {number|null} probability
 * @property {number|null} low
 * @property {number|null} high
 * @property {number|null} majority_label
 * @property {'estimated'|'uncertain'|'insufficient'|'gold'} status
 * @property {string} reason
 */
/**
 * @typedef {Object} CalibrationAnnotator
 * @property {string} annotator
 * @property {number} opinions
 * @property {number|null} sensitivity
 * @property {number|null} specificity
 * @property {number|null} false_negative_rate
 * @property {number|null} false_positive_rate
 * @property {'reliable'|'unstable'|'reversed'|'insufficient'|'unanchored'} status
 */
/**
 * @typedef {Object} Calibration
 * @property {number} id
 * @property {string} name
 * @property {'queued'|'running'|'done'|'failed'} state
 * @property {number} attempts
 * @property {string} error
 * @property {{annotators:CalibrationAnnotator[],claims:CalibrationClaim[],checks:object[],evidence:object}} [result]
 */
