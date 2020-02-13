package test;

import org.junit.Test;

public class MyTest {

	@Test
	public void test() {
		long availMemory = Runtime.getRuntime().maxMemory();
		long requiredMemory = 1024 * 1024 * 1024;
		if (availMemory <  requiredMemory) {
			throw new OutOfMemoryError("Memory is only " + (availMemory / 1024 / 1024) + "MB");
		}
	}

}
