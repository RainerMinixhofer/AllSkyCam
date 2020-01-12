.PHONY: clean All

All:
	@echo "----------Building project:[ DHT - Release ]----------"
	@cd "DHT" && "$(MAKE)" -f  "DHT.mk"
clean:
	@echo "----------Cleaning project:[ DHT - Release ]----------"
	@cd "DHT" && "$(MAKE)" -f  "DHT.mk" clean
