create database library_system;
use library_system;
SET GLOBAL local_infile=ON;
create table books (
	book_id int primary key, 
    title varchar(255),
    author varchar(255), 
    edition int, 
    pub_year year
);
load data local infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/books.csv' 
into table books 
fields terminated by ',' 
enclosed by '"' 
ignore 1 rows;
select * from books;
create table courses (
	course_code varchar(20) primary key, 
    course_name varchar(255)
);
load data local infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/courses.csv' into table courses fields terminated by ',' enclosed by '"' ignore 1 rows;
select * from courses;
create table branch_course (
	course_code varchar(20),
    branch varchar(10), 
    semester int, 
    PRIMARY KEY (course_code, branch, semester),
    foreign key (course_code) references courses(course_code)
);
load data local infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/branch-course.csv' into table branch_course fields terminated by ',' enclosed by '"' ignore 1 rows;
select * from branch_course;
create table courses_books (
	course_code varchar(20), 
	book_id int, 
	PRIMARY KEY (course_code, book_id),
    FOREIGN KEY (course_code) REFERENCES courses(course_code),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
load data local infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/courses-books.csv' into table courses_books fields terminated by ',' enclosed by '"' ignore 1 rows;
select * from courses_books;
delete from courses_books where book_id = 0;