import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;

public class DatasetUtil {
	
		// Example for parameter stringRange
		// "first-3","5","6-last"
		public static Instances removeAttributes(String stringRange, Instances instances) throws Exception {
			Remove remove = new Remove();
			remove.setAttributeIndices(stringRange);
			remove.setInputFormat(instances);
			return Filter.useFilter(instances, remove);
		}
		
		public static Instances randomize(Instances instances) throws Exception {
			Filter filter = new Randomize();
			filter.setInputFormat(instances);
			return Filter.useFilter(instances,filter);
		}
		
		//Need to be tested
		public static Instances resampling(Instances instances) throws Exception {
			instances = randomize(instances);
			if (instances != null) {
				Resample resample = new Resample();
				resample.setBiasToUniformClass(1.0);
				resample.setSampleSizePercent(100.0);
				resample.setInputFormat(instances);
				resample.setRandomSeed((int) (Integer.MAX_VALUE * Math.random())); 
				return Filter.useFilter(instances, resample); 
			}
			return null;
		}
}
