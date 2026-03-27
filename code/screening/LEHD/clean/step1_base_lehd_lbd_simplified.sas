/*********************************************************/
/* AGGREGATE IMPLICATIONS OF MERGERS FOR US LABOR MARKET */
/* ----------------------------------------------------- */
/* STEP 1. CREATING LEHD-LBD BASE FILE ----------------- */ 
/*********************************************************/

/* In this file, we create an establishment-year panel from the LEHD files
	- In essence, we create a LBD-like from the LEHD files.

	- We begin by importing files from the ECF and transformation them into a plant year 
	  panel structure, following the LBD.

	- From here, we fill out missing firmid information using EIN linkages through the revised LBD.
	
	- Lastly, we merge in commuting zone information for later analysis.
*/

/* Begin by defining relevant paths */
libname PthIn "";
libname PthOut "";
libname lehd "";
libname temp "";

/* Defining system options */
options obs=max;

/* Importing county to cz crosswalk */
proc import out=PthIn.county_cz_all_crosswalk
	datafile=""
	dbms=dta
	replace;
run;

/* Utilizing macro environment to iterate through full 28 states */
%macro raw_state_loop;

	/* List of state abbreviations */
	%let list = md al az co ct de ia in ks ma me nd nj nm nv ny 
		oh ok pa sc sd tn tx ut va wa wi wy;

	/* Initializing min and max year variables */
	%let firstyear = 2001;
	%let lastyear = 2018;
	%let firstyearmin = &firstyear. - 1;
	%let secondlastyear = &lastyear. - 1;

	/* Initializing loop counter */
	%let i = 1;

	/* Iterating through all states to obtain ECF files */
	%do %while (%scan(&list, &i) ne );

		/* Setting 'next_state' to be the proceeding state abbreviation in list */
        	%let next_state = %scan(&list, &i);

		/***************************************************/
		/*   1. IMPORT FILES AND SAVE RELEVANT VARIABLES   */
		/***************************************************/

		/* Directories for state-level employer characteristic files */
        	libname ecfdir "";
		libname et26dir "";

		/* Merging ECF-T26 information into ECF-SEINUNIT file */
		proc sql;
			create table temp.lehd_lbd_&next_state._base as 

			select A.sein, B.seinunit, A.year, A.quarter, A.firmid, A.fas_ein, 
				A.fas_ein_flag, A.lbd_match, A.multi_unit_lbd, 
				B.in_202, B.in_ui, B.es_state, B.leg_county, B.leg_cbsa, 
				B.leg_cbsa_memi, B.naics2017fnl, B.best_emp1, B.best_emp2,
				B.best_emp3, B.best_wages

			from et26dir.ecf_&next_state._sein_t26 as A left join
				ecfdir.ecf_&next_state._seinunit as B

			on A.sein = B.sein AND 
				A.year = B.year AND
				A.quarter = B.quarter;
		quit;

		/* Keeping data before 2001 (ie 2000) so we can identify
			firm ownership changes in 2001 itself */
		data temp.lehd_lbd_&next_state._base;
			set temp.lehd_lbd_&next_state._base;
			if year < &firstyearmin. then delete;
			if year > &lastyear. then delete;
			keep sein seinunit year quarter firmid fas_ein fas_ein_flag lbd_match
				multi_unit_lbd in_202 in_ui best_emp: best_wages es_state
				leg_county leg_cbsa leg_cbsa_memi naics2017fnl;
			rename es_state=fipsst;
			rename leg_county=fipscou;
			rename leg_cbsa=cbsa;
			rename leg_cbsa_memi=cbsa_f;
			rename naics2017fnl=fknaics2017;
		run;

		/**************************************************************/
		/*   2. CONVERT TO PLANT-YEAR PANEL FOLLOWING LBD STRUCTURE   */
		/**************************************************************/

		/* Sorting data by SEIN, SEINUNIT, year, and firmid, note that we sort 
			by firmid in descending order so missings are last, rather than first */
		proc sort data=temp.lehd_lbd_&next_state._base
			out=temp.lehd_lbd_&next_state._base;
			by sein seinunit year descending firmid;
		run;

		/* It is possible that firmid is not available for every year-quarter pair, 
			so if it is available within the same year for another quarter then 
			we take that entry for firmid instead */
		data temp.lehd_lbd_&next_state._base;
			set temp.lehd_lbd_&next_state._base;
			by sein seinunit year;
			retain firmid_fill;
			if first.year=1 then firmid_fill=firmid;
		run;

		proc sort data=temp.lehd_lbd_&next_state._base
			out=temp.lehd_lbd_&next_state._base;
			by sein seinunit year quarter;
		run;

		/* Within the LEHD, we take the 3rd month of the first quarter (ie March) */
		data temp.lehd_lbd_&next_state._base;
			set temp.lehd_lbd_&next_state._base;
			firmid_final = firmid;
			if firmid="" & quarter=1 then firmid_final=firmid_fill;
			emp_lehd=best_emp3;
		run;

		/* Generating annual payroll */
		proc sql;
			create table temp.lehd_lbd_&next_state._base as 

			select A.*, sum(A.best_wages) as pay_lehd
			from temp.lehd_lbd_&next_state._base as A
			group by sein, seinunit, year;
		quit;

		/* Reducing to annual panel. First we create an indicator variable 
			within each SEIN-SEINUNIT-year level for whether the 
			first quarter is present. */
		data temp.lehd_lbd_&next_state._base;
			set temp.lehd_lbd_&next_state._base;
			d_qrt1=0;
			quarter_keep=0;
			if quarter=1 then d_qrt1=1;
			if quarter=1 then quarter_keep=1;
		run;

		proc sql;
			create table temp.lehd_lbd_&next_state._base as 
			
			select A.*, sum(A.d_qrt1) as qrt1
			from temp.lehd_lbd_&next_state._base as A
			group by sein, seinunit, year;
		quit;

		data temp.lehd_lbd_&next_state._base;
			set temp.lehd_lbd_&next_state._base;
			lehdnum=cat(sein, seinunit);
			if quarter>1 & pay_lehd>0 & qrt1=0 then quarter_keep=1;
			if quarter_keep=1 then output;
			if quarter~=1 then emp_lehd=0;
			drop firmid;
			keep sein seinunit lehdnum year firmid_final fas_ein fas_ein_flag lbd_match
				multi_unit_lbd in_202 in_ui emp_lehd pay_lehd fipsst fipscou 
				cbsa cbsa_f cblibname lbdrevsa_f fknaics2017;
			rename firmid_final=firmid;
		run;

		/* Sorting by SEIN, SEINUNIT, year and removing duplicates */
		proc sort data=temp.lehd_lbd_&next_state._base
			out=temp.lehd_lbd_&next_state._base nodupkey;
			by sein seinunit year;
		run;

		/**********************************************************/
		/*   3. APPENDING STATE-LEVEL FILES INTO NATIONAL PANEL   */
		/**********************************************************/

		%if &i.=1 %then 
			%do;
				data PthOut.lehd_lbd_allstates_base_raw;
					set temp.lehd_lbd_&next_state._base;
				run;
			%end;
		%else 
			%do;
				proc append base=PthOut.lehd_lbd_allstates_base_raw
					data=temp.lehd_lbd_&next_state._base;
				run;
			%end;

		/* Preparing raw base file for merge with ein-firmid lists */
		data PthOut.lehd_lbd_allstates_base_raw;
			set PthOut.lehd_lbd_allstates_base_raw;
			ein_5d=substr(fas_ein, 1, 5);
			ein=substr(fas_ein, 6, 9);
		run;

		/* Deleting now unneeded state level file */
		proc datasets library=temp nolist;
			delete lehd_lbd_&next_state._base;
		run;

		/* Incrementing loop counter */
       	 	%let i = %eval(&i+1);
	%end;

%mend raw_state_loop;
%raw_state_loop;

/* Employing macro environment to iterate over all years in our sample */
%macro raw_year_loop;

	/* Initializing min and max year variables */
	%let firstyear = 2001;
	%let lastyear = 2018;
	%let firstyearmin = %eval(&firstyear. - 1);
	%let secondlastyear = %eval(&lastyear. - 1);

	/*****************************************************************************/
	/*   4. IMPROVING EIN-FIRMID LINKAGES IN ECF-T26 FILES AND ADDING "LBDFID"   */
	/*****************************************************************************/

	%do year = &firstyearmin. %to &lastyear.;

		/* Directory for revised LBD-E files */
		libname lbde clear;
		libname lbde "";

		data temp.ein_firmid_&year._list;
			set lbde.lbd&year._v201900;
			if firmid="" then delete;
			if ein="" then delete;
			keep year ein firmid lbdfid;
			rename firmid=firmid_lbd;
			rename lbdfid=lbdfid_lbd;
		run;

		/* Sorting at the EIN level and removing duplicates */
		proc sort data=temp.ein_firmid_&year._list
			out=temp.ein_firmid_&year._list nodupkey;
			by ein;
		run;

		/* Appending each yearly file into one master file */
		%if &year.=&firstyearmin. %then 
			%do;
				data temp.ein_firmid_allyears_list;
					set temp.ein_firmid_&year._list;
				run;
			%end;
		%else
			%do;
				proc append base=temp.ein_firmid_allyears_list
					data=temp.ein_firmid_&year._list;
				run;
			%end;

		/* Deleting now unnecessary yearly file */
		proc datasets library=temp nolist;
			delete ein_firmid_&year._list;
		run;
	%end;

	/* Soring all years file at the EIN, year level */
	proc sort data=temp.ein_firmid_allyears_list
		out=temp.ein_firmid_allyears_list;
		by ein year;
	run;

	/* Merging potential additional EIN-firmid linkages */
	proc sql;
		create table PthOut.lehd_lbd_allstates_base as 

		select A.*, B.firmid_lbd, B.lbdfid_lbd 

		from PthOut.lehd_lbd_allstates_base_raw as A
		left join temp.ein_firmid_allyears_list as B

		on A.year = B.year AND
		A.ein = B.ein AND
		~missing(A.year) AND
		~missing(B.year);
	quit;

	/* Finding additional firmids */
	data PthOut.lehd_lbd_allstates_base;
		set PthOut.lehd_lbd_allstates_base;
		firmid_final = firmid;
		if ein_5d="EINUS" & firmid="" & firmid_lbd~="" then firmid_final=firmid_lbd;
		drop firmid_lbd firmid;
		rename firmid_final=firmid;
		rename lbdfid_lbd=lbdfid;
	run;

	/* Deleting already used file */
	proc datasets library=temp nolist;
		delete ein_firmid_allyears_list;
	run;
	
%mend raw_year_loop;
%raw_year_loop;
 
/******************************************/
/*   5. MERGE IN COMMUTING ZONES (CZs)    */
/******************************************/

proc sql;
	create table PthOut.lehd_lbd_allstates_CZ as

	select A.*, B.fipscou_orig, B.cz_id2000, B.cz_id1990, B.cz_id1980, B.county_name,
		B.msa_2003_name, B.county_pop_2003, B.cz_pop_2000

	from PthOut.lehd_lbd_allstates_base as A left join
		PthIn.county_cz_all_crosswalk as B

	on A.fipscou=B.fipscou AND
		~missing(A.fipscou);
quit;

 



