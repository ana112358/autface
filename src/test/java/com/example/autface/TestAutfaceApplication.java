package com.example.autface;

import org.springframework.boot.SpringApplication;

public class TestAutfaceApplication {

	public static void main(String[] args) {
		SpringApplication.from(AutfaceApplication::main).with(TestcontainersConfiguration.class).run(args);
	}

}
