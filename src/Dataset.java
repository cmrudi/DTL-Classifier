import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Dataset {
	private Instances instances;
	
	public Dataset(String pathFile) throws Exception {
		DataSource dataSource = new DataSource(pathFile);
		this.instances = dataSource.getDataSet();
	}
	
	public void updateInstances(Instances instances) {
		this.instances = instances;
	}
	
	public Id3 buildId3(Instances instances) throws Exception {
		this.instances.setClassIndex(instances.numAttributes() - 1);
		Id3 id3 = new Id3();
		id3.buildClassifier(this.instances);
		return id3;
	}
	
	public J48 buildJ48(Instances instances) throws Exception {
		this.instances.setClassIndex(instances.numAttributes() - 1);
		J48 j48 = new J48();
		j48.buildClassifier(this.instances);
		return j48;
	}
	
	
	
	
	
}
