/*********************************************************/
/* AGGREGATE IMPLICATIONS OF MERGERS FOR US LABOR MARKET */
/* ----------------------------------------------------- */
/* STEP 7. ESTABLISHMENT-LEVEL REVENUES ---------------- */ 
/*********************************************************/

/* In this step, we calculate establishment-year level revenues. To do so, we perform the following steps:
 	a. Pull/generate EIN-level revenues from SSL/BR files
		1. For 2001, we use the variables "acsr1-acsr4" and follow the flow chart in Appendix A (page 21) of Haltiwanger et al. (August 2020)
		2. For 2002-2016, we use the variable "bestadmin_rcpt_xxxx"
		3. For 2017-2018, we only have revenues at the "firmid" level; pull variable "nrev2" from the "brfirm_xxxx" files
 	b. Determine share of employment for each state at the EIN (2001-2016) and "firmid" (2017-2018) level
   In step 10, we distribute revenues across plants within a state according to employment
*/

/* Begin by defining our paths. */
libname LBDpath "";
libname LEHDpath "";
libname model "";

/*********************************************************/
/***   GENERATING PLANT-LEVEL REVENUES FOR 2001-2018   ***/
/*********************************************************/

%macro revenue_construction;

	/* Iterating through each of the years in our data range */
	%do year = 2001 %to 2018;

		/* Assigning directory of SSL/BR data */
		libname ssl_&year. "";
		libname lbd_&year. "";

		/* DELETE STATEMENT FOR INDIVIDUAL SSL YEARS */
		proc datasets library=model nolist;
			delete orig_br_&year.
		     	       br_&year._revenues
		       	       br_firmid_revs_&year.;
		run;

		/* Importing BR file depending on the year */
		/* - We have a specific process for calculating revenues in 
		   the year 2001, which is not necessary in later years.
	  	 - This process comes from the appendix of the .pdf file: 
	 	  /data/economics/ssl/doc/brfirm_rev_documentation_v2016.pdf 
	 	  - This file will also be referred to as Haltiwanger et al. (August 2020)
		*/

		/* 2001: Follow definitions from Haltiwanger et al. (August 2020) */
		%if &year. = 2001 %then %do;
		
			data model.br_&year._revenues;
				set ssl_&year..ssl&year.su;

				/* IRS Form Flag matching indicator */
				if acsr1f=acsr2f and acsr2f=acsr3f and acsr3f=acsr4f 
					then equal_ind = 1;
				else equal_ind = 0;

				/* Sector definition - first two digits of NAICS variable "admnaics"*/
				sector = substr(admnaics, 1, 2);

				/* Direct calculation of revenues/receipts using given flowchart */
				if acsr1f="" or acsr1f="D" then revenues=acsr1;

				else if sector in ("52", "53", "55") then do;	
					if acsr1f in ("1", "2", "3", "4", "5", "E") then do;
						revenues=sum(of acsr1-acsr3);
					end;

					else if acsr1f in ("6", "7") then do;
						revenues=sum(acsr1,acsr2);
					end;

					else if acsr1f in ("8", "9", "A", "B", "C") then do;
						revenues=acsr1;
					end;
				end;
				
				else if sector in ("51", "54", "56", "61", "62", "71", "81") then do;	
					if acsr1f in ("1", "2", "3", "4", "5", "8", "B", "C", "E") then do;
						revenues=acsr1;
					end;

					else if acsr1f in ("9", "A") then do;
						revenues=sum(of acsr1-acsr3);
					end;
				end;
				
				else if sector in ("21", "23", "31", "33", "42", "44", "45", "72") then do;
					if acsr1f in ("1", "2", "3", "4", "5", "8", "C", "E", "B") then do;
						revenues=acsr1;
					end;

					else if acsr1f in ("9", "A") then do;
						revenues=sum(acsr1,acsr3);
					end;
				end;
				
				else if acsr1f in ("6", "7") then revenues=.;
					else revenues=acsr1;

			/* Keep relevant variables */
			retain year &year.;
			keep ein revenues year;

			run;

		%end;

		/* 2002-2016: revenues are readily available in variable "bestadmin_rcpt_xxxx" */
		%if &year. > 2001 and &year. < 2017 %then %do;
			data model.br_&year._revenues;
				set ssl_&year..ssl&year.su;

				/* Keep relevant variables */
				retain year &year.;
				keep ein bestadmin_rcpt_&year. year;
				rename bestadmin_rcpt_&year. = revenues;
			run;
		%end;

		/* 2017-2018: revenues are available at the "firmid" level in variable "nrev2" */
		%if &year. > 2016 %then %do;
			data model.br_&year._revenues;
				set lbd_&year..brfirm_rev&year._v201900;

				/* Prep for revenue distribution by employment weights */
				retain year &year.;
				keep firmid nrev2 year;
				rename nrev2 = revenues;
			run;
		%end;

	%end;

%mend revenue_construction;
%revenue_construction;

/* Append revenue files across two periods:
	- Period 1: 2001 - 2016
	- Period 2: 2017 - 2018
*/

/* Period 1: 2001 - 2016 */
%macro revenue_append;

	/* Iterating through each of the years in our data range */
	%do year = 2001 %to 2016;
		
		/* Append across years */
		%if &year. = 2001 %then %do;
			data model.br_2001_2016_revenues;
				set model.br_&year._revenues;
			run;
		%end;

		%if &year. > 2001 %then %do;
			proc append base=model.br_2001_2016_revenues
			  	    data=model.br_&year._revenues;
			run;
		%end;

		/* Delete temporary files to preserve HD space */
		proc datasets library=model nolist;
			delete br_&year._revenues;
		run;

	%end;

%mend revenue_append;
%revenue_append;

/* Period 2: 2017 - 2018 */
data model.br_2017_2018_revenues;
	set model.br_2017_revenues;
run;

proc append base=model.br_2017_2018_revenues
	    data=model.br_2018_revenues;
run;

/* Delete temporary files to preserve HD space */
proc datasets library=model nolist;
	delete br_2017_revenues
	       br_2018_revenues;
run;	

/* b. Determine share of employment for each state at the EIN (2001-2016) and firmid (2017-2018) level 
	- We pull this information from the LBD
*/

%macro state_weights;

	/* Iterating through each of the years in our data range */
	%do year = 2001 %to 2018;

		/* Assigning directory of revised LBD data */
		libname lrdir "";
		libname model "";
	
		/* 2001 - 2016 */
		%if &year.<2017 %then %do;

			/* Determine total employment at EIN-state level */
			proc sql;
				create table model.state_rev_weights1_&year. as
				select ein, bds_st, year, sum(emp) as emp_upper_st
	       			from lrdir.lbd&year._v201900
	       			group by ein, bds_st;
			quit;

			proc sort data=model.state_rev_weights1_&year.
				out=model.state_rev_weights1_&year.
				NODUPKEY;
				by ein bds_st;
			run;

			/* Determine total employment at EIN level */
			proc sql;
				create table model.state_rev_weights2_&year. as
				select ein, year, sum(emp) as emp_upper
       				from lrdir.lbd&year._v201900
        			group by ein;
			quit;

			proc sort data=model.state_rev_weights2_&year.
				out=model.state_rev_weights2_&year.
				NODUPKEY;
				by ein;
			run;

			/* Create data set with employment at EIN-state and EIN level */
			proc sql;
				create table model.state_rev_weights_&year. as

				select A.*, B.emp_upper

				from model.state_rev_weights1_&year. as A,
            		    	     model.state_rev_weights2_&year. as B

				where A.ein=B.ein 
			    	      and ~missing(A.ein)
		                      and ~missing(B.ein);
			quit;

		%end;

		/* 2017 - 2018 */
		%if &year.>2016 %then %do;

			/* Determine total employment at firmid-state level */
			proc sql;
				create table model.state_rev_weights1_&year. as
				select firmid, bds_st, year, sum(emp) as emp_upper_st
       				from lrdir.lbd&year._v201900
       				group by firmid, bds_st;
			quit;

			proc sort data=model.state_rev_weights1_&year.
				out=model.state_rev_weights1_&year.
				NODUPKEY;
				by firmid bds_st;
			run;

			/* Determine total employment at firmid level */
			proc sql;
				create table model.state_rev_weights2_&year. as
				select firmid, year, sum(emp) as emp_upper
       				from lrdir.lbd&year._v201900
        			group by firmid;
			quit;

			proc sort data=model.state_rev_weights2_&year.
				out=model.state_rev_weights2_&year.
				NODUPKEY;
				by firmid;
			run;

			/* Create data set with employment at firmid-state and firmid level */
			proc sql;
				create table model.state_rev_weights_&year. as

				select A.*, B.emp_upper

				from model.state_rev_weights1_&year. as A,
            		     	     model.state_rev_weights2_&year. as B

				where A.firmid=B.firmid 
			    	      and ~missing(A.firmid)
		            	      and ~missing(B.firmid);
			quit;

		%end;

		/* Append across years (2001-2016) */
		%if &year. = 2001 %then %do;
			data model.state_rev_weights_allyears1;
				set model.state_rev_weights_&year.;
			run;
		%end;

		%if &year. > 2001 and &year.<2017 %then %do;
			proc append base=model.state_rev_weights_allyears1
			  	    data=model.state_rev_weights_&year.;
			run;
		%end;	

		/* Delete temporary files to preserve HD space */
		%if &year.<2017 %then %do;		
			proc datasets library=model nolist;
				delete state_rev_weights1_&year.
		      	     	       state_rev_weights2_&year.
		              	       state_rev_weights_&year.;
			run;
		%end;	

	%end;

%mend state_weights;
%state_weights;

/* Append across years (2017-2018) */
data model.state_rev_weights_allyears2;
	set model.state_rev_weights_2017;
run;

proc append base=model.state_rev_weights_allyears2
	    data=model.state_rev_weights_2018;
run;

proc datasets library=model nolist;
	delete state_rev_weights1_2017
	       state_rev_weights2_2017
	       state_rev_weights_2017
	       state_rev_weights1_2018
	       state_rev_weights2_2018
	       state_rev_weights_2018;
run;

