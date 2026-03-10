/*********************************************************/
/* AGGREGATE IMPLICATIONS OF MERGERS FOR US LABOR MARKET */
/* ----------------------------------------------------- */
/* STEP 4. CREATING FULL JOB HISTORY FOR EACH INDIVIDUAL */
/* WHO HAS EVER WORKED AT ESTABLISHMENT INVOLVED WITH -- */
/* A "LARGE" LOCAL MERGER ------------------------------ */ 
/*********************************************************/

/* In this step, we calculate the full job history of each individual who
   has ever worked for an establishment associated with a "large" local merger
	- This list of establishments is created in step 3
	- The full job history of an individual is, of course, limited by the set of states we have available
*/

/* Begin by defining our paths. */
libname LBDpath "";
libname LEHDpath "";

/* Declaring macro to loop over all states in our sample */
%macro stateloop;

    /* Full list of states and counter for the loop:
       "proc append" cannot create variables, so we need to put the state with the most "e" variables in the loop first;
       this is Maryland (MD) - e21 */
    %let list = md al az co ct de ia in ks ma me nd nj nm nv ny oh ok pa sc sd tn tx ut va wa wi wy;
    %local i next_state;

    /* Initializing loop counter at 1 */
    %let i = 1;

    /* Loop itself */
    %do %while (%scan(&list, &i) ne );

        /* Setting 'next_state' to be the proceeding state abbreviation in list */
        %let next_state = %scan(&list, &i);

        /* Directory for state-level job history files (JHF) */
        libname jhfdir "";

        /* Generate list of individuals who were employed by top merger firms at some point in time 
        Note: - This merge identifies only PIK-SEIN pairs in which the SEIN is associated with a top merger firm
        */
        proc sql;
                create table LEHDpath.step4_&next_state._piklist as

                /* Selecting only needed variables from JHF files */
                select A.*
            
                    /* Utilizing previously generated list of all SEINs involved 
                        in the top 100 mergers, in conjunction with our JHF files. */
                from jhfdir.jhf_&next_state. (keep = pik sein) as A,
                LEHDpath.&next_state._seinlist as B

                    /* This is key, it says to only consider JHF workers with the SEINs from our LBD top merger file */
                where A.sein=B.sein and ~missing(A.sein) and ~missing(B.sein);
        quit;

        /* Get rid of duplicates in PIK and obtain list of PIKs involved with top merger firms at SOME point in time */
            proc sort data=LEHDpath.step4_&next_state._piklist(keep=pik) out=LEHDpath.step4_&next_state._piklist_final NODUPKEY;
                by _ALL_;
            run;

        /* Find COMPLETE/FULL job history of individuals who were EVER involved with a top merger firm 
        Note: - This merge allows us to track whether PIKs employed at top 100 merger firms ever went to ANY other firm
        */
        proc sql;
                create table LEHDpath.step4_&next_state._full_job_history as

                /* Selecting only needed variables from JHF files */
                select A.*, max(A.last_sep) as last_sep_max, min(A.first_acc) as first_acc_min
            
                    /* Utilizing previously generated list of all PIKs involved 
                        in the top 100 mergers, in conjunction with our JHF files. */
                from jhfdir.jhf_&next_state. (keep = pik sein seinunit1 fid spell_u2w first_acc last_sep) as A,
                LEHDpath.step4_&next_state._piklist_final as B

                    /* This is key, it says to only consider a worker's job history if it was ever employed by a top merger firm */
                where A.pik=B.pik and ~missing(A.pik) and ~missing(B.pik)

            /* Grouping by PIK-spell */
                group by A.pik, A.fid, A.spell_u2w;
        quit;

        /* Sort data on relevant variables */
        proc sort data=LEHDpath.step4_&next_state._full_job_history;
            by pik fid spell_u2w first_acc last_sep;
        run;

        /* Clean up newly-created state-level files:
        - create new variables through concatenation (for "plant" and "spell")
        - create state indicator
        - collapse to one observation per PIK-spell episode
        - delete observations with missing values */
        data LEHDpath.step4_&next_state._full_job_history_clean;
            set LEHDpath.step4_&next_state._full_job_history;
            by pik fid spell_u2w;

            plant=catx('_', sein, seinunit1);
            state_source = "&next_state";

            retain plant_first;
            retain plant_last;

	    /*
            if first.pik and first.fid and first.spell_u2w then plant_first = plant;
            if last.pik and last.fid and last.spell_u2w then plant_last = plant;

            if first.pik and first.fid and first.spell_u2w then output;
	    */

	    if first.spell_u2w then plant_first = plant;
            if last.spell_u2w then plant_last = plant;

            if first.spell_u2w then output;

            keep pik state_source plant_first plant_last first_acc_min last_sep_max;
        run;

        /* Deleting temporary state-level dataset to preserve memory. */
        proc datasets library=LEHDpath nolist;
            delete step4_&next_state._piklist step4_&next_state._piklist_final step4_&next_state._full_job_history;
        run;

	/* Collapse ECF file containing location and industry information 
	- There is no need for this information to be varying over year-quarter pairs
	*/
	libname ecfdir "";
	
	data LEHDpath.ecf_&next_state._seinunit_geo_ind;
            set ecfdir.ecf_&next_state._seinunit;
	    plant_first=catx('_', sein, seinunit);
	    keep plant_first sein seinunit leg_cbsa leg_county naics2017fnl;
	run;

	 /* Get rid of duplicates in SEIN-SEINUNIT */
        proc sort data=LEHDpath.ecf_&next_state._seinunit_geo_ind out=LEHDpath.ecf_&next_state._seinunit_geo_ind2 NODUPKEY;
            by sein seinunit;
        run;

	/* ADDITION: MERGE IN CZ AT THE SEIN SEINUNIT LEVEL */
	proc sql;

		create table LEHDpath.cz_sein_seinunit_merge as		

		select A.sein, A.seinunit, A.cz_id2000, B.usps_code

		from LBDpath.all_mergers_list as A,
		LBDpath.cz_to_usps_code as B

		where A.cz_id2000 = B.cz_id2000 AND 
			~missing(A.cz_id2000) AND 
			~missing(B.cz_id2000);
	quit;

	data LEHDpath.cz_sein_seinunit_merge;
		set LEHDpath.cz_sein_seinunit_merge;
		if usps_code ~= "&next_state." then delete;
	run;

	proc sql;
		create table LEHDpath.ecf_&next_state._seinunit_geo_ind2 as
		
		select A.*, B.cz_id2000

		from LEHDpath.ecf_&next_state._seinunit_geo_ind2 as A,
		LEHDpath.cz_sein_seinunit_merge as B

		where A.sein = B.sein AND 
			A.seinunit = B.seinunit AND 
			~missing(A.sein) AND 
			~missing(B.sein) AND 
			~missing(A.seinunit) AND
			~missing(B.seinunit);
	quit;
	
	/* Merge geographic and industry information at SEIN-SEINUNIT level into state-level full job history files */
	proc sql;
            create table LEHDpath.step4_&next_state._full_job_history_clean2 as

            select *
            from LEHDpath.step4_&next_state._full_job_history_clean as A,
            LEHDpath.ecf_&next_state._seinunit_geo_ind2 as B

            where A.plant_first=B.plant_first and ~missing(A.plant_first) and ~missing(B.plant_first);
        quit;

        /* Initializing full state dataset for the first state we look at */
        %if &i = 1 %then %do;
            data LEHDpath.step4_full_job_history_all2;
                set LEHDpath.step4_&next_state._full_job_history_clean2;
            run;
        %end;

        /* For all states other than the first, appending merged dataset to the 
            already initialized full state data */
        %if &i > 1 %then %do; 
            proc append base = LEHDpath.step4_full_job_history_all2
		        data = LEHDpath.step4_&next_state._full_job_history_clean2;
            run;
        %end;
        
        /* Incrementing loop counter */
        %let i = %eval(&i+1);
        
    %end;

%mend stateloop;
%stateloop;

/* Sort file containing full job history of PIKs associated with top merger firms */
proc sort data=LEHDpath.step4_full_job_history_all2;
	by pik first_acc_min last_sep_max state_source;
run;

/* Create job sequence variable - this allows us to determine the "degree" of connections later */
data LEHDpath.step4_full_job_history_all2;
	set LEHDpath.step4_full_job_history_all2;
	
	by pik;
	if first.pik
	then job_seq = 1;
	else job_seq +1;
run;
	



